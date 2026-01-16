// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
#include <algorithm>
#include <vector>
#include <cmath>

// --- Utilitaires ---
// log-sum-exp pour stabilité numérique
double log_sum_exp(const vec &v) {
  double m = v.max();
  return m + std::log(sum(exp(v - m)));
}

// calcule densité log d'une gaussienne multivariée (full cov) : log N(x | mu, Sigma)
// pour efficience on suppose Sigma est covariance et precalc inverse + logdet sont donnés
double log_mvnorm(const vec &x, const vec &mu, const mat &invSig, double logdetSigma, int d) {
  vec diff = x - mu;
  double val = -0.5 * ( d * std::log(2.0 * M_PI) + logdetSigma + as_scalar(diff.t() * invSig * diff) );
  return val;
}

// --- EM pour HMM gaussien multivarié ---
// Inputs:
//   X: (T x D) observations (rows are time points)
//   nstates: number of hidden states
//   maxit, tol
// Outputs: a list with pi (init probs), A (transition matrix), mu (k x d), Sigma_inv, logdetSigma, gamma, xi, logLik, etc.
// For robust numeric behavior we regularize covariances with small diag add (eps)
struct HMMModel {
  vec pi; // init probs (K)
  mat A;  // KxK transitions
  mat mu; // K x D means (each row a mean)
  std::vector<mat> Sigma; // K covariance matrices (DxD)
  std::vector<mat> Sigma_inv; // inverses
  vec logdetSigma; // K entries
  double logLik;
};

// EM training
HMMModel train_hmm_em(const mat &X, int K, int maxit=200, double tol=1e-6, double cov_reg = 1e-6, bool verbose=false) {
  int T = X.n_rows;
  int D = X.n_cols;
  // initialize with K-means for means (simple)
  mat centers;
  {
    // use arma's kmeans (may fail parfois) fallback to random
    bool ok = true;
    try {
      centers.set_size(D, K);
      if (!kmeans(centers, X.t(), K, random_subset, 10, false)) {
        ok = false;
      }
    } catch(...) { ok = false; }
    if (!ok) {
      // random partition
      centers.zeros();
      for (int i=0;i<T;i++) centers.col(i % K) += X.row(i).t();
      for (int k=0;k<K;k++) centers.col(k) /= (double)std::max(1,(int)floor((double)T/K));
    }
  }
  // convert centers to mu rows (K x D)
  mat mu(K, D);
  for (int k=0;k<K;k++) mu.row(k) = centers.col(k).t();

  // initialize pi uniform, A near uniform with slight diag preference
  vec pi = vec(K); pi.fill(1.0 / K);
  mat A(K,K); A.fill(1.0 / K);
  for (int k=0;k<K;k++) A(k,k) += 0.1;
  for (int k=0;k<K;k++) A.row(k) /= accu(A.row(k));

  // initial covariances: global covariance or per-cluster sample cov
  std::vector<mat> Sigma(K);
  mat globalCov = cov(X);
  for (int k=0;k<K;k++) {
    Sigma[k] = globalCov;
    Sigma[k] += cov_reg * eye<mat>(D,D);
  }

  // precalc inverse and logdet
  std::vector<mat> Sigma_inv(K);
  vec logdet(K);
  for (int k=0;k<K;k++) {
    Sigma_inv[k] = inv_sympd(Sigma[k]);
    logdet(k) = std::log(det(Sigma[k]) + 1e-20);
  }

  double prevLogLik = -datum::inf;
  double logLik = 0.0;

  // EM iterations
  mat emission_logprob(T, K);
  for (int iter=0; iter<maxit; ++iter) {
    // E-step: compute emission_logprob[t,k] = log p(x_t | state=k)
    for (int k=0;k<K;k++) {
      // parallelize over t OR parallelize outer k?
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int t=0;t<T;t++) {
        // avoid creating vec x inside inner loop? we still need x each t
        vec x = X.row(t).t();
        emission_logprob(t,k) = log_mvnorm(x, mu.row(k).t(), Sigma_inv[k], logdet(k), D);
      }
    }
    // Forward-backward in log-space
    // alpha_log[t,k], beta_log[t,k]
    mat alpha_log(T, K);
    mat beta_log(T, K);
    // initial alpha: log(pi) + emission at t=0
    for (int k=0;k<K;k++) alpha_log(0,k) = std::log(std::max(1e-300, pi(k))) + emission_logprob(0,k);
    // forward
    for (int t=1;t<T;t++) {
      for (int j=0;j<K;j++) {
        vec tmp(K);
        for (int i=0;i<K;i++) tmp(i) = alpha_log(t-1,i) + std::log(std::max(1e-300, A(i,j)));
        alpha_log(t,j) = log_sum_exp(tmp) + emission_logprob(t,j);
      }
    }
    // backward
    for (int k=0;k<K;k++) beta_log(T-1,k) = 0.0;
    for (int t=T-2;t>=0;--t) {
      for (int i=0;i<K;i++) {
        vec tmp(K);
        for (int j=0;j<K;j++) tmp(j) = std::log(std::max(1e-300, A(i,j))) + emission_logprob(t+1,j) + beta_log(t+1,j);
        beta_log(t,i) = log_sum_exp(tmp);
      }
    }
    // gamma (posterior state probs) and xi (pairwise)
    mat gamma(T, K);
    std::vector<mat> xi(T-1); // each is KxK
    for (int t=0;t<T;t++) {
      vec v(K);
      for (int k=0;k<K;k++) v(k) = alpha_log(t,k) + beta_log(t,k);
      double denom = log_sum_exp(v);
      for (int k=0;k<K;k++) gamma(t,k) = std::exp(v(k) - denom);
    }
    for (int t=0;t<T-1;t++) {
      mat M(K,K);
      // Build matrix of log-probabilities for pairs (i,j)
      for (int i=0;i<K;i++) {
        for (int j=0;j<K;j++) {
          double val = alpha_log(t,i) + std::log(std::max(1e-300, A(i,j))) + emission_logprob(t+1,j) + beta_log(t+1,j);
          M(i,j) = val;
        }
      }
      // normalize using log-sum-exp over all i,j
      vec flat(K*K);
      int idx=0;
      for (int i=0;i<K;i++) for (int j=0;j<K;j++) flat(idx++) = M(i,j);
      double log_denom = log_sum_exp(flat);
      xi[t].set_size(K,K);
      for (int i=0;i<K;i++) for (int j=0;j<K;j++) xi[t](i,j) = std::exp(M(i,j) - log_denom);
    }

    // compute log-likelihood as log-sum-exp of last alpha_log row
    vec last = alpha_log.row(T-1).t();
    logLik = log_sum_exp(last);

    // M-step: update pi, A, mu, Sigma
    // pi = gamma[0]
    for (int k=0;k<K;k++) pi(k) = std::max(1e-12, gamma(0,k));
    pi /= accu(pi);

    // A: sum_t xi / sum_t gamma_t
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
      // normalize row
      double rowSum = accu(A.row(i));
      if (rowSum <= 0) {
        A.row(i).fill(1.0/K);
      } else {
        A.row(i) /= rowSum;
      }
    }

    // mu: weighted average by gamma
    for (int k=0;k<K;k++) {
      vec num(D); num.zeros();
      double den = 0.0;
      for (int t=0;t<T;t++) {
        num += gamma(t,k) * X.row(t).t();
        den += gamma(t,k);
      }
      if (den <= 0) mu.row(k).zeros();
      else mu.row(k) = (num / den).t();
    }

    // Sigma: weighted covariance
    for (int k=0;k<K;k++) {
      mat S(D,D); S.zeros();
      double den = 0.0;
      for (int t=0;t<T;t++) {
        vec diff = X.row(t).t() - mu.row(k).t();
        S += gamma(t,k) * (diff * diff.t());
        den += gamma(t,k);
      }
      if (den <= 0) {
        Sigma[k] = globalCov + cov_reg*eye<mat>(D,D);
      } else {
        Sigma[k] = S / den;
        Sigma[k] += cov_reg * eye<mat>(D,D);
      }
      Sigma_inv[k] = inv_sympd(Sigma[k]);
      logdet(k) = std::log(det(Sigma[k]) + 1e-20);
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
  out.Sigma = Sigma;
  out.Sigma_inv = Sigma_inv;
  out.logdetSigma = logdet;
  out.logLik = logLik;
  return out;
}

// Viterbi decoding: given HMMModel and data X (T x D), return state sequence (1..K)
// Uses precalculated inverses in model
uvec viterbi_decode(const HMMModel &mdl, const mat &X) {
  int T = X.n_rows;
  int K = mdl.pi.n_elem;
  int D = X.n_cols;
  mat delta(T, K);
  umat psi(T, K);
  // t=0
  for (int k=0;k<K;k++) {
    double logpi = std::log(std::max(1e-300, mdl.pi(k)));
    double logem = log_mvnorm(X.row(0).t(), mdl.mu.row(k).t(), mdl.Sigma_inv[k], mdl.logdetSigma(k), D);
    delta(0,k) = logpi + logem;
    psi(0,k) = 0;
  }
  for (int t=1;t<T;t++) {
    for (int j=0;j<K;j++) {
      vec tmp(K);
      for (int i=0;i<K;i++) tmp(i) = delta(t-1,i) + std::log(std::max(1e-300, mdl.A(i,j)));
      uword imax = tmp.index_max();
      double maxv = tmp(imax);
      double logem = log_mvnorm(X.row(t).t(), mdl.mu.row(j).t(), mdl.Sigma_inv[j], mdl.logdetSigma(j), D);
      delta(t,j) = maxv + logem;
      psi(t,j) = imax;
    }
  }
  // backtrack
  uvec states(T);
  states(T-1) = delta.row(T-1).t().index_max();
  for (int t=T-2;t>=0;--t) states(t) = psi(t+1, states(t+1));
  // convert to 1-based for R compatibility
  for (int t=0;t<T;t++) states(t) = states(t) + 1;
  return states;
}


// quantile_nth function

// prototype pour Rcpp::NumericVector version (si définie ailleurs)
double quantile_nth(Rcpp::NumericVector xv, double p);

// overload pour arma::vec
double quantile_nth(const arma::vec & xv, double p) {
  if (xv.n_elem == 0) Rcpp::stop("empty vector");
  if (p < 0.0 || p > 1.0) Rcpp::stop("probability p must be in [0,1]");

  // copier dans std::vector<double> (pour utiliser nth_element in-place)
  std::vector<double> x;
  x.reserve(xv.n_elem);
  for (arma::uword i = 0; i < xv.n_elem; ++i) x.push_back(xv(i));

  size_t n = x.size();

  if (p == 0.0) return *std::min_element(x.begin(), x.end());
  if (p == 1.0) return *std::max_element(x.begin(), x.end());

  // simulate R's type=7: h = 1 + (n - 1) * p
  double h = 1.0 + (static_cast<double>(n) - 1.0) * p;
  double hf = std::floor(h);
  double frac = h - hf;
  size_t i = static_cast<size_t>(hf); // 1-based index
  size_t k = (i == 0) ? 0 : (i - 1);  // 0-based
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


// --- Walk-forward driver ---
// Inputs:
//   X_all: T x D feature matrix (no NAs), aligned to the tail of original price series
//   nstates, n_bull, n_bear, mode_select ("mean","mean_sd","percentile"), percentile_cut
//   training_frequency, initial_multiplier, seed
// Output: List with signals (T ints -1/0/1), states (T ints 1..K or NA-coded 0), diagnostics per task
// For numeric stability and reproducibility seed used to initialize random subset for kmeans
//' Entraînement HMM multivarié en walk-forward et génération de signaux
//'
//' Entraîne un Hidden Markov Model (HMM) multivarié par EM sur fenêtres glissantes,
//' décode les états avec l'algorithme de Viterbi et produit des signaux -1/0/1.
//'
//' Cette fonction est une interface Rcpp/C++ pour un usage embarqué dans un package R.
//' Elle suppose que la première colonne de X_all contient les rendements (ret) utilisés
//' pour scorer les états (les autres colonnes peuvent contenir ATR, RSI, etc.).
//'
//' @param X_all numeric matrix (T x D). Matrice de features alignée avec la série de prix.
//'        Aucune valeur manquante autorisée. Première colonne : rendements (ret).
//' @param nstates integer (>=1). Nombre d'états cachés K. Default: 2.
//' @param n_bull integer. Nombre d'états considérés comme "bull". Default: 1.
//' @param n_bear integer. Nombre d'états considérés comme "bear". Default: 1.
//' @param mode_select character. Méthode de sélection des états: "mean", "mean_sd" ou "percentile".
//'        Default: "mean".
//' @param percentile_cut numeric (0..0.5). Seuil pour la sélection par percentile (si mode_select == "percentile").
//'        Default: 0.2.
//' @param seed integer. Seed de initialisation (affecte kmeans initial). Default: 123.
//' @param training_frequency integer. Fréquence (en pas) entre ré-entrainements (taille de la fenêtre OOS).
//'        Default: 21.
//' @param initial_multiplier integer. Taille de la fenêtre initiale = initial_multiplier * training_frequency.
//'        Default: 3.
//' @param maxit integer. Nombre maximal d'itérations EM. Default: 200.
//' @param tol numeric. Tolérance de convergence EM (relatif sur logLik). Default: 1e-6.
//' @param verbose logical. Messages de progression si TRUE. Default: TRUE.
//'
//' @return A named list with elements:
//'   \item{signals}{integer vector length T (-1: bear, 0: neutral, 1: bull)}'
//'   \item{states}{integer vector length T with decoded states (1..K) or 0 for unassigned}'
//'   \item{diagnostics}{list of per-task diagnostics; each element contains state_means, state_sds,
//'         bull_states, bear_states, train_end, predict_end, oos_counts}'
//'
//' @details
//' - La fonction entraîne le HMM sur une fenêtre initiale, puis re-entraine périodiquement (walk-forward).
//' - L'algorithme EM utilise une régularisation sur les covariances (cov_reg = 1e-6).
//' - La sélection des états bull / bear privilégie les états observés dans la période out-of-sample.
//' - La colonne 1 de X_all doit être les rendements utilisés pour scorer les états.
//'
//' @examples
//' \dontrun{
//' library(Rcpp)
//' # Supposons que vous ayez un package 'votrePackage' et que la DLL soit chargée
//' # Exemple synthétique : T x D matrix
//' set.seed(1)
//' T <- 200
//' D <- 4
//' X_all <- matrix(rnorm(T * D), ncol = D)
//' # Appel direct (si la fonction est exportée dans votre package)
//' res <- votrePackage::walk_forward_hmm_cpp(X_all, nstates = 3, training_frequency = 20, verbose = FALSE)
//' str(res)
//' }
//'
//' @references
//' - Rabiner, L. (1989) A tutorial on Hidden Markov Models and selected applications in speech recognition.
//' - Bishop, C. M. (2006) Pattern Recognition and Machine Learning (EM, GMM).
//'
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

   // compute ends
   std::vector<int> ends;
   for (int e = initial_window; e <= T_all; e += training_frequency) ends.push_back(e);
   if (ends.back() != T_all) ends.push_back(T_all);

   // tasks are pairs (train_end, predict_end) for ends[i-1] -> ends[i]
   int ntasks = (int)ends.size() - 1;
   if (ntasks <= 0) stop("No tasks created");

   arma::ivec signal(T_all); signal.fill(0);
   arma::ivec states(T_all); states.fill(0); // 0 means NA/unassigned
   List diagnostics(ntasks);

   for (int ti = 0; ti < ntasks; ++ti) {
     int train_end = ends[ti];
     int predict_end = ends[ti+1];
     // build data matrices
     mat X_train = X_all.rows(0, train_end-1); // 0-based indices, inclusive
     mat X_allwin = X_all.rows(0, predict_end-1);

     // train HMM on X_train
     // set seed variation
     std::srand(seed + ti);
     HMMModel model = train_hmm_em(X_train, nstates, maxit, tol, 1e-6, false);

     // decode on X_allwin
     uvec states_all = viterbi_decode(model, X_allwin); // 1..K
     // compute state-level stats using ret (assume ret is first column) — caller must ensure column order ret, atr, rsi, macdh
     vec ret_vals = X_allwin.col(0); // use the whole window (train+oos) per original R code
     // compute mean and sd for each state
     vec state_means(nstates); state_means.fill(NA_REAL);
     vec state_sds(nstates); state_sds.fill(NA_REAL);
     for (int k=1;k<=nstates;k++) {
       uvec idx = find(states_all == (uword)k);
       if (idx.n_elem > 0) {
         vec vals(idx.n_elem);
         for (uword j=0;j<idx.n_elem;j++) vals(j) = ret_vals(idx(j));
         state_means(k-1) = mean(vals);
         state_sds(k-1) = (vals.n_elem > 1) ? stddev(vals) : 0.0;
       } else {
         state_means(k-1) = NA_REAL;
         state_sds(k-1) = NA_REAL;
       }
     }
     // scoring and selection of bull/bear states
     vec score = state_means;
     if (mode_select == "mean_sd") {
       for (int k=0;k<nstates;k++) {
         double s = state_sds(k);
         if (std::isnan(state_means(k)) || std::isnan(s) || s <= 1e-12) score(k) = NA_REAL;
         else score(k) = state_means(k) / s;
       }
     }

     // --- Robust selection patch starts here ---
     // Order states by score descending (handle NA by sending to end)
     std::vector<int> ord_states(nstates);
     iota(ord_states.begin(), ord_states.end(), 1); // 1..K
     std::sort(ord_states.begin(), ord_states.end(), [&](int a, int b){
       double sa = score(a-1), sb = score(b-1);
       if (std::isfinite(sa) && std::isfinite(sb)) return sa > sb;
       if (std::isfinite(sa) && !std::isfinite(sb)) return true;
       if (!std::isfinite(sa) && std::isfinite(sb)) return false;
       return a < b;
     });

     // prepare from/to indices for OOS portion (zero-based)
     int from = train_end;
     int to = predict_end - 1;

     // count occurrences in OOS for each state
     vec oos_counts(nstates); oos_counts.zeros();
     if (to >= from) {
       for (int t = from; t <= to; ++t) {
         int st = (int) states_all(t); // 1..K
         if (st >= 1 && st <= nstates) oos_counts(st-1) += 1;
       }
     }

     std::vector<int> bull_states;
     std::vector<int> bear_states;

     if (mode_select != "percentile") {
       // choose bull_states: top n_bull prioritizing states present in OOS
       for (size_t i=0; i<ord_states.size() && (int)bull_states.size() < n_bull; ++i) {
         int s = ord_states[i];
         if (oos_counts(s-1) > 0) bull_states.push_back(s);
       }
       // backfill from ord_states if not enough present in OOS
       for (size_t i=0; i<ord_states.size() && (int)bull_states.size() < n_bull; ++i) {
         int s = ord_states[i];
         if (std::find(bull_states.begin(), bull_states.end(), s) == bull_states.end()) bull_states.push_back(s);
       }

       // choose bear_states: bottom n_bear prioritizing states present in OOS
       for (int i=(int)ord_states.size()-1; i>=0 && (int)bear_states.size() < n_bear; --i) {
         int s = ord_states[i];
         if (oos_counts(s-1) > 0) bear_states.push_back(s);
       }
       // backfill if needed
       for (int i=(int)ord_states.size()-1; i>=0 && (int)bear_states.size() < n_bear; --i) {
         int s = ord_states[i];
         if (std::find(bear_states.begin(), bear_states.end(), s) == bear_states.end()) bear_states.push_back(s);
       }
     } else {
       // percentile selection on state_means (existing logic) but prefer states present in OOS
       vec valid_means;
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
       // ensure at least n_bull/n_bear using priority for OOS presence and backfill from ord_states
       // bulls
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
     // --- Robust selection patch ends here ---

     // assign signals for out-of-sample portion (train_end .. predict_end-1) zero-based
     int from_assign = train_end; int to_assign = predict_end - 1;
     for (int t=from_assign; t<=to_assign; ++t) {
       int st = (int)states_all(t); // states_all is 1-based
       if (std::find(bull_states.begin(), bull_states.end(), st) != bull_states.end()) signal(t) = 1;
       else if (std::find(bear_states.begin(), bear_states.end(), st) != bear_states.end()) signal(t) = -1;
       else signal(t) = 0;
       states(t) = st;
     }

     // diagnostics
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
