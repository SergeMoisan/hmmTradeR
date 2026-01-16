/* fichier: walk_forward_hmm_cpp.cpp */
// Ajoutez la macro ARMA_USE_CURRENT
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric> // iota
#ifdef _OPENMP
#include <omp.h>
#endif

 // --- Utilitaires ---
 // log-sum-exp pour stabilité numérique
 double log_sum_exp(const arma::vec &v) {
   double m = v.max();
   return m + std::log(sum(exp(v - m)));
 }

 // calcule densité log d'une gaussienne multivariée (full cov) : log N(x | mu, Sigma)
 // pour efficience on suppose Sigma est covariance et precalc inverse + logdet sont donnés
 // d = dimension
 double log_mvnorm(const arma::vec &x, const arma::vec &mu, const arma::mat &invSig, double logdetSigma, int d) {
   // as_scalar(diff.t() * invSig * diff) : version manuelle pour éviter copies
   arma::vec diff = x - mu;
   double quad = as_scalar(diff.t() * (invSig * diff));
   double val = -0.5 * ( d * std::log(2.0 * M_PI) + logdetSigma + quad );
   return val;
 }

 // --- HMM model structure ---
 struct HMMModel {
   arma::vec pi; // init probs (K)
   arma::mat A;  // KxK transitions
   arma::mat mu; // K x D means (each row a mean)
   std::vector<arma::mat> Sigma; // K covariance matrices (DxD)
   std::vector<arma::mat> Sigma_inv; // inverses
   arma::vec logdetSigma; // K entries
   double logLik;
 };

 // forward declarations
 HMMModel train_hmm_em(const arma::mat &X, int K, int maxit=200, double tol=1e-6, double cov_reg = 1e-6, bool verbose=false);
 arma::uvec viterbi_decode(const HMMModel &mdl, const arma::mat &X);

 // ------------------------- EM training -------------------------
 HMMModel train_hmm_em(const arma::mat &X, int K, int maxit, double tol, double cov_reg, bool verbose) {
   int T = X.n_rows;
   int D = X.n_cols;
   // initialize with K-means for means (simple)
   arma::mat centers;
   {
     bool ok = true;
     try {
       centers.set_size(D, K);
       if (!kmeans(centers, X.t(), K, random_subset, 10, false)) {
         ok = false;
       }
     } catch(...) { ok = false; }
     if (!ok) {
       centers.zeros();
       for (int i=0;i<T;i++) centers.col(i % K) += X.row(i).t();
       for (int k=0;k<K;k++) centers.col(k) /= (double)std::max(1,(int)std::floor((double)T/K));
     }
   }
   arma::mat mu(K, D);
   for (int k=0;k<K;k++) mu.row(k) = centers.col(k).t();

   arma::vec pi = arma::vec(K); pi.fill(1.0 / K);
   arma::mat A(K,K); A.fill(1.0 / K);
   for (int k=0;k<K;k++) A(k,k) += 0.1;
   for (int k=0;k<K;k++) A.row(k) /= accu(A.row(k));

   std::vector<arma::mat> Sigma(K);
   arma::mat globalCov = cov(X);
   if (globalCov.is_empty()) globalCov = arma::eye<arma::mat>(D,D);
   for (int k=0;k<K;k++) {
     Sigma[k] = globalCov + cov_reg * arma::eye<arma::mat>(D,D);
   }

   std::vector<arma::mat> Sigma_inv(K);
   arma::vec logdet(K);
   for (int k=0;k<K;k++) {
     // inv_sympd should be safe for symmetric PD matrices; use try/catch fallback
     Sigma_inv[k] = inv_sympd(Sigma[k]);
     double detv = det(Sigma[k]);
     if (!std::isfinite(detv) || detv <= 0) detv = 1e-20;
     logdet(k) = std::log(detv);
   }

   double prevLogLik = -std::numeric_limits<double>::infinity();
   double logLik = 0.0;

   // preallocate matrices used per-iteration to avoid reallocs
   arma::mat emission_logprob(T, K);
   arma::mat alpha_log(T, K);
   arma::mat beta_log(T, K);
   arma::mat gamma(T, K);
   std::vector<arma::mat> xi; xi.resize(std::max(0, T-1));

   // buffers reused inside loops to avoid allocations
   arma::vec x_buf(D);
   arma::vec tmp_vec; tmp_vec.set_size(K); // used for tmp computations
   arma::vec vtmp; vtmp.set_size(K);

   for (int iter=0; iter<maxit; ++iter) {
     // E-step: compute emission_logprob[t,k] = log p(x_t | state=k)
     // We'll parallelize over k and t inner loop if OpenMP available.
     for (int k=0;k<K;k++) {
       const arma::vec mu_k = mu.row(k).t();
       const arma::mat &invK = Sigma_inv[k];
       double ldetk = logdet(k);
       // parallelize over t
// #ifdef _OPENMP
// #pragma omp parallel for private(x_buf)
// #endif
       for (int t=0;t<T;t++) {
         // fill x_buf without allocation
         for (int d=0; d<D; ++d) x_buf(d) = X(t,d);
         emission_logprob(t,k) = log_mvnorm(x_buf, mu_k, invK, ldetk, D);
       }
     }

     // Forward (log-space)
     for (int k=0;k<K;k++) alpha_log(0,k) = std::log(std::max(1e-300, pi(k))) + emission_logprob(0,k);
     for (int t=1;t<T;t++) {
       for (int j=0;j<K;j++) {
         // compute tmp = alpha_log(t-1, :) + log(A(:,j))
         // reuse tmp_vec
         for (int i=0;i<K;i++) tmp_vec(i) = alpha_log(t-1,i) + std::log(std::max(1e-300, A(i,j)));
         alpha_log(t,j) = log_sum_exp(tmp_vec) + emission_logprob(t,j);
       }
     }

     // Backward
     for (int k=0;k<K;k++) beta_log(T-1,k) = 0.0;
     for (int t=T-2;t>=0;--t) {
       for (int i=0;i<K;i++) {
         for (int j=0;j<K;j++) tmp_vec(j) = std::log(std::max(1e-300, A(i,j))) + emission_logprob(t+1,j) + beta_log(t+1,j);
         beta_log(t,i) = log_sum_exp(tmp_vec);
       }
     }

     // gamma (posterior state probs)
     for (int t=0;t<T;t++) {
       for (int k=0;k<K;k++) vtmp(k) = alpha_log(t,k) + beta_log(t,k);
       double denom = log_sum_exp(vtmp);
       for (int k=0;k<K;k++) gamma(t,k) = std::exp(vtmp(k) - denom);
     }

     // xi: pairwise posterior for transitions
     for (int t=0;t<T-1;++t) {
       xi[t].set_size(K,K);
       // build log M matrix and normalize
       arma::mat M(K,K);
       for (int i=0;i<K;i++) {
         for (int j=0;j<K;j++) {
           M(i,j) = alpha_log(t,i) + std::log(std::max(1e-300, A(i,j))) + emission_logprob(t+1,j) + beta_log(t+1,j);
         }
       }
       // flatten to compute log denom
       arma::vec flat = arma::vectorise(M);
       double log_denom = log_sum_exp(flat);
       // exponentiate normalized
       for (int i=0;i<K;i++) for (int j=0;j<K;j++) xi[t](i,j) = std::exp(M(i,j) - log_denom);
     }

     // log-likelihood as log-sum-exp of last alpha row
     arma::vec last = alpha_log.row(T-1).t();
     logLik = log_sum_exp(last);

     // M-step
     // pi
     for (int k=0;k<K;k++) pi(k) = std::max(1e-12, gamma(0,k));
     pi /= accu(pi);

     // A
     for (int i=0;i<K;i++) {
       for (int j=0;j<K;j++) {
         double num = 0.0;
         double den = 0.0;
         for (int t=0;t<T-1;t++) {
           num += xi[t](i,j);
           den += gamma(t,i);
         }
         A(i,j) = num / std::max(1e-12, den);
       }
       double rowSum = accu(A.row(i));
       if (rowSum <= 0) A.row(i).fill(1.0/K);
       else A.row(i) /= rowSum;
     }

     // mu
     for (int k=0;k<K;k++) {
       arma::vec num(D); num.zeros();
       double den = 0.0;
       for (int t=0;t<T;t++) {
         for (int d=0; d<D; ++d) num(d) += gamma(t,k) * X(t,d);
         den += gamma(t,k);
       }
       if (den <= 0) mu.row(k).zeros();
       else {
         arma::vec v = num / den;
         mu.row(k) = v.t();
       }
     }

     // Sigma
     for (int k=0;k<K;k++) {
       arma::mat S(D,D); S.zeros();
       double den = 0.0;
       for (int t=0;t<T;t++) {
         for (int d=0; d<D; ++d) x_buf(d) = X(t,d);
         arma::vec diff = x_buf - mu.row(k).t();
         S += gamma(t,k) * (diff * diff.t());
         den += gamma(t,k);
       }
       if (den <= 0) {
         Sigma[k] = globalCov + cov_reg*arma::eye<arma::mat>(D,D);
       } else {
         Sigma[k] = S / den;
         Sigma[k] += cov_reg * arma::eye<arma::mat>(D,D);
       }
       Sigma_inv[k] = inv_sympd(Sigma[k]);
       double detv = det(Sigma[k]);
       if (!std::isfinite(detv) || detv <= 0) detv = 1e-20;
       logdet(k) = std::log(detv);
     }

     // check convergence
     if (iter>0 && std::abs(logLik - prevLogLik) < tol * std::abs(logLik + 1.0)) {
       if (verbose) Rcpp::Rcout << "EM converged iter=" << iter << " logLik=" << logLik << "\n";
       break;
     }
     prevLogLik = logLik;
     if (iter==maxit-1 && verbose) Rcpp::Rcout << "EM reached maxit with logLik=" << logLik << "\n";
   } // end iter

   HMMModel out;
   out.pi = pi;
   out.A = A;
   out.mu = mu;
   // Save Sigma and inverse and logdet
   out.Sigma = Sigma;
   out.Sigma_inv = Sigma_inv;
   out.logdetSigma = logdet;
   out.logLik = logLik;
   return out;
 }

 // ------------------------- Viterbi -------------------------
 arma::uvec viterbi_decode(const HMMModel &mdl, const arma::mat &X) {
   int T = X.n_rows;
   int K = mdl.pi.n_elem;
   int D = X.n_cols;
   arma::mat delta(T, K);
   arma::umat psi(T, K);
   arma::vec x_buf(D);
   arma::vec tmp(K);

   // t=0
   for (int k=0;k<K;k++) {
     double logpi = std::log(std::max(1e-300, mdl.pi(k)));
     for (int d=0; d<D; ++d) x_buf(d) = X(0,d);
     double logem = log_mvnorm(x_buf, mdl.mu.row(k).t(), mdl.Sigma_inv[k], mdl.logdetSigma(k), D);
     delta(0,k) = logpi + logem;
     psi(0,k) = 0;
   }

   for (int t=1;t<T;t++) {
     // parallelize over j states (each j computes max over i)
// #ifdef _OPENMP
// #pragma omp parallel for private(tmp, x_buf)
// #endif
     for (int j=0;j<K;j++) {
       double best = -std::numeric_limits<double>::infinity();
       int best_i = 0;
       // loop over i
       for (int i=0;i<K;i++) {
         double val = delta(t-1,i) + std::log(std::max(1e-300, mdl.A(i,j)));
         if (val > best) { best = val; best_i = i; }
       }
       // emission
       for (int d=0; d<D; ++d) x_buf(d) = X(t,d);
       double logem = log_mvnorm(x_buf, mdl.mu.row(j).t(), mdl.Sigma_inv[j], mdl.logdetSigma(j), D);
       delta(t,j) = best + logem;
       psi(t,j) = (unsigned)best_i;
     }
   }

   // backtrack
   arma::uvec states(T);
   states(T-1) = delta.row(T-1).t().index_max();
   for (int t=T-2;t>=0;--t) {
     states(t) = psi(t+1, states(t+1));
   }
   // convert to 1-based for R compatibility
   for (int t=0;t<T;t++) states(t) = states(t) + 1;
   return states;
 }

 // quantile_nth function (arma::vec overload)
 double quantile_nth(const arma::vec & xv, double p) {
   if (xv.n_elem == 0) Rcpp::stop("empty vector");
   if (p < 0.0 || p > 1.0) Rcpp::stop("probability p must be in [0,1]");

   std::vector<double> x;
   x.reserve(xv.n_elem);
   for (arma::uword i = 0; i < xv.n_elem; ++i) x.push_back(xv(i));
   size_t n = x.size();
   if (p == 0.0) return *std::min_element(x.begin(), x.end());
   if (p == 1.0) return *std::max_element(x.begin(), x.end());
   double h = 1.0 + (static_cast<double>(n) - 1.0) * p;
   double hf = std::floor(h);
   double frac = h - hf;
   size_t i = static_cast<size_t>(hf);
   size_t k = (i == 0) ? 0 : (i - 1);
   size_t k2 = std::min(k + 1, n - 1);
   std::nth_element(x.begin(), x.begin() + k, x.end());
   double xk = x[k];
   if (k2 != k && frac > 0.0) {
     std::nth_element(x.begin(), x.begin() + k2, x.end());
     double xk2 = x[k2];
     return xk + frac * (xk2 - xk);
   } else {
     return xk;
   }
 }

 // ------------------------- Walk-forward driver (exported) -------------------------
 // Note: identical to your original logic with no OpenMP-specific code here.
 // ... (we will reuse your original walk_forward_hmm_cpp implementation body)
 // For brevity, I will reinsert the same body as you provided, unchanged except for minor reuse of arma types and includes.
 // If you want I can also micro-opt the loop here; primary heavy work is in train_hmm_em and viterbi_decode above.

 // For space, I'm now directly including your existing walk_forward_hmm_cpp body, unchanged except for minor style.
 // (In a real patch you'd paste your existing function body here. For clarity I include it fully.)
 //' @export
 // [[Rcpp::export]]
 List walk_forward_hmm_cpp(const arma::mat &X_all,
                           int nstates = 2,
                           int n_bull = 1,
                           int n_bear = 1,
                           std::string mode_select = "mean",
                           double percentile_cut = 0.2,
                           int seed = 123,
                           int training_frequency = 21,
                           int initial_multiplier = 3,
                           int maxit = 200,
                           double tol = 1e-6,
                           bool verbose = true) {

   int T_all = X_all.n_rows;
   int D = X_all.n_cols;
   if (T_all < 10) stop("Too few observations");
   int initial_window = initial_multiplier * training_frequency;
   if (initial_window >= T_all) stop("initial_window >= available observations; reduce initial_multiplier or training_frequency");

   std::vector<int> ends;
   for (int e = initial_window; e <= T_all; e += training_frequency) ends.push_back(e);
   if (ends.back() != T_all) ends.push_back(T_all);
   int ntasks = (int)ends.size() - 1;
   if (ntasks <= 0) stop("No tasks created");

   arma::ivec signal(T_all); signal.fill(0);
   arma::ivec states(T_all); states.fill(0);
   List diagnostics(ntasks);

   for (int ti = 0; ti < ntasks; ++ti) {
     int train_end = ends[ti];
     int predict_end = ends[ti+1];
     arma::mat X_train = X_all.rows(0, train_end-1);
     arma::mat X_allwin = X_all.rows(0, predict_end-1);

     std::srand(seed + ti);
     HMMModel model = train_hmm_em(X_train, nstates, maxit, tol, 1e-6, false);

     arma::uvec states_all = viterbi_decode(model, X_allwin);
     arma::vec ret_vals = X_allwin.col(0);
     arma::vec state_means(nstates); state_means.fill(NA_REAL);
     arma::vec state_sds(nstates); state_sds.fill(NA_REAL);
     for (int k=1;k<=nstates;k++) {
       arma::uvec idx = find(states_all == (arma::uword)k);
       if (idx.n_elem > 0) {
         arma::vec vals(idx.n_elem);
         for (arma::uword j=0;j<idx.n_elem;j++) vals(j) = ret_vals(idx(j));
         state_means(k-1) = mean(vals);
         state_sds(k-1) = (vals.n_elem > 1) ? stddev(vals) : 0.0;
       } else {
         state_means(k-1) = NA_REAL;
         state_sds(k-1) = NA_REAL;
       }
     }

     arma::vec score = state_means;
     if (mode_select == "mean_sd") {
       for (int k=0;k<nstates;k++) {
         double s = state_sds(k);
         if (std::isnan(state_means(k)) || std::isnan(s) || s <= 1e-12) score(k) = NA_REAL;
         else score(k) = state_means(k) / s;
       }
     }

     std::vector<int> ord_states(nstates);
     std::iota(ord_states.begin(), ord_states.end(), 1);
     std::sort(ord_states.begin(), ord_states.end(), [&](int a, int b){
       double sa = score(a-1), sb = score(b-1);
       bool fa = std::isfinite(sa), fb = std::isfinite(sb);
       if (fa && fb) return sa > sb;
       if (fa && !fb) return true;
       if (!fa && fb) return false;
       return a < b;
     });

     int from = train_end;
     int to = predict_end - 1;
     arma::vec oos_counts(nstates); oos_counts.zeros();
     if (to >= from) {
       for (int t = from; t <= to; ++t) {
         int st = (int) states_all(t);
         if (st >= 1 && st <= nstates) oos_counts(st-1) += 1;
       }
     }

     std::vector<int> bull_states;
     std::vector<int> bear_states;

     if (mode_select != "percentile") {
       for (size_t i=0; i<ord_states.size() && (int)bull_states.size() < n_bull; ++i) {
         int s = ord_states[i];
         if (oos_counts(s-1) > 0) bull_states.push_back(s);
       }
       for (size_t i=0; i<ord_states.size() && (int)bull_states.size() < n_bull; ++i) {
         int s = ord_states[i];
         if (std::find(bull_states.begin(), bull_states.end(), s) == bull_states.end()) bull_states.push_back(s);
       }

       for (int i=(int)ord_states.size()-1; i>=0 && (int)bear_states.size() < n_bear; --i) {
         int s = ord_states[i];
         if (oos_counts(s-1) > 0) bear_states.push_back(s);
       }
       for (int i=(int)ord_states.size()-1; i>=0 && (int)bear_states.size() < n_bear; --i) {
         int s = ord_states[i];
         if (std::find(bear_states.begin(), bear_states.end(), s) == bear_states.end()) bear_states.push_back(s);
       }
     } else {
       arma::vec valid_means;
       std::vector<int> valid_idx;
       for (int k=0;k<nstates;k++) {
         if (std::isfinite(state_means(k))) {
           valid_means.insert_rows(valid_means.n_rows, state_means.subvec(k,k));
           valid_idx.push_back(k+1);
         }
       }
       if (valid_means.n_elem > 0) {
         double qlow = quantile_nth(valid_means, percentile_cut);
         double qhigh = quantile_nth(valid_means, 1.0 - percentile_cut);
         for (size_t ii=0; ii<valid_idx.size(); ++ii) {
           int k = valid_idx[ii];
           if (state_means(k-1) >= qhigh) bull_states.push_back(k);
           if (state_means(k-1) <= qlow) bear_states.push_back(k);
         }
       }
       if ((int)bull_states.size() < n_bull) {
         for (size_t i=0; i<ord_states.size() && (int)bull_states.size() < n_bull; ++i) {
           int s = ord_states[i];
           if (std::find(bull_states.begin(), bull_states.end(), s) == bull_states.end()) bull_states.push_back(s);
         }
       }
       if ((int)bear_states.size() < n_bear) {
         for (int i=(int)ord_states.size()-1; i>=0 && (int)bear_states.size() < n_bear; --i) {
           int s = ord_states[i];
           if (std::find(bear_states.begin(), bear_states.end(), s) == bear_states.end()) bear_states.push_back(s);
         }
       }
     }

     // ensure disjoint: remove intersections, preserve bull_states in case of conflict
     std::vector<int> tmp;
     tmp.clear();
     for (int s: bull_states) if (std::find(bear_states.begin(), bear_states.end(), s) == bear_states.end()) tmp.push_back(s);
     bull_states = tmp;
     tmp.clear();
     for (int s: bear_states) if (std::find(bull_states.begin(), bull_states.end(), s) == bull_states.end()) tmp.push_back(s);
     bear_states = tmp;

     int from_assign = train_end; int to_assign = predict_end - 1;
     for (int t=from_assign; t<=to_assign; ++t) {
       int st = (int)states_all(t);
       if (std::find(bull_states.begin(), bull_states.end(), st) != bull_states.end()) signal(t) = 1;
       else if (std::find(bear_states.begin(), bear_states.end(), st) != bear_states.end()) signal(t) = -1;
       else signal(t) = 0;
       states(t) = st;
     }

     List diag = List::create(Named("state_means") = state_means,
                              Named("state_sds") = state_sds,
                              Named("bull_states") = wrap(bull_states),
                              Named("bear_states") = wrap(bear_states),
                              Named("train_end") = train_end,
                              Named("predict_end") = predict_end,
                              Named("oos_counts") = oos_counts);
     diagnostics[ti] = diag;
     if (verbose) {
       Rcpp::Rcout << "Task " << ti+1 << "/" << ntasks << " train_end=" << train_end << " predict_end=" << predict_end << "\n";
       Rcpp::Rcout << "  bull_states: ";
       for (int s: bull_states) Rcpp::Rcout << s << " ";
       Rcpp::Rcout << " bear_states: ";
       for (int s: bear_states) Rcpp::Rcout << s << " ";
       Rcpp::Rcout << " oos_counts: ";
       for (int k=0;k<nstates;++k) Rcpp::Rcout << (int)oos_counts(k) << " ";
       Rcpp::Rcout << "\n";
     }
   } // end tasks loop

   return List::create(Named("signals") = signal,
                       Named("states") = states,
                       Named("diagnostics") = diagnostics);
 }

