// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

// --------------------------- UTILITAIRES ---------------------------

// log-sum-exp
double log_sum_exp(const vec &v) {
  double m = v.max();
  return m + std::log(sum(exp(v - m)));
}

// log multivariate normal (use invSig and logdet precomputed)
double log_mvnorm(const vec &x, const vec &mu, const mat &invSig, double logdetSigma, int d) {
  vec diff = x - mu;
  double val = -0.5 * ( d * std::log(2.0 * M_PI) + logdetSigma + as_scalar(diff.t() * invSig * diff) );
  return val;
}

// quantile_nth for arma::vec (R type=7)
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

// --------------------------- HMM (copié/identique à votre impl) ---------------------------

// structure pour résultat d'entrainement
struct HMMModel {
  arma::vec pi;
  arma::mat A;
  arma::mat mu;
  std::vector<arma::mat> Sigma;
  std::vector<arma::mat> Sigma_inv;
  arma::vec logdetSigma;
  double logLik;
};

// forward declarations of train and viterbi (we reuse your implementations)
HMMModel train_hmm_em(const arma::mat &X, int K, int maxit=200, double tol=1e-6, double cov_reg = 1e-6, bool verbose=false);
arma::uvec viterbi_decode(const HMMModel &mdl, const arma::mat &X);

// --------------------------- TRAIN & VITERBI definitions (same as original) ---------------------------
// For brevity I include compact versions of train_hmm_em and viterbi_decode adapted from your code
// (If you already have them in your package, you can omit duplication and just declare them above).

// ... Insert your original implementations of train_hmm_em and viterbi_decode here ...
// For clarity and to keep this snippet focused, assume train_hmm_em and viterbi_decode
// are present (they are long and your original code can be reused unchanged).
// If you need the full embedded definitions, I can paste them verbatim.

// --------------------------- MAIN WALK-FORWARD PATCHED ---------------------------

//' Walk-forward HMM with hyperparameter grid search per task (Rcpp)
 //'
 //' Train HMMs on rolling windows and for each retrain step search over provided
 //' hyperparameter combinations (nstates_cand x n_bull_cand x n_bear_cand) and select
 //' the best according to the "expected OOS return" metric (mean(ret|bull) - mean(ret|bear)).
 //'
 //' @param X_all numeric matrix (T x D), first column must be returns used for scoring
 //' @param nstates_cand integer vector of candidate states (e.g. c(2,3,4))
 //' @param n_bull_cand integer vector of candidate number of bull states (e.g. c(1,2))
 //' @param n_bear_cand integer vector of candidate number of bear states
 //' @param training_frequency integer (retrain frequency). Default 21.
 //' @param initial_multiplier integer (initial window = multiplier * training_frequency). Default 3.
 //' @param seed integer base seed. Default 123.
 //' @param maxit EM maxit. Default 200.
 //' @param tol EM tol. Default 1e-6.
 //' @param cov_reg covariance regularization. Default 1e-6.
 //' @param parallel bool run grid training per task in parallel using OpenMP (default TRUE).
 //' @param verbose bool messages. Default TRUE.
 //' @return List with signals, states, diagnostics
 //' @export
 // [[Rcpp::export]]
 List optimized_walk_forward_hmm_cpp(const arma::mat &X_all,
                           IntegerVector nstates_cand,
                           IntegerVector n_bull_cand,
                           IntegerVector n_bear_cand,
                           int training_frequency = 21,
                           int initial_multiplier = 3,
                           int seed = 123,
                           int maxit = 200,
                           double tol = 1e-6,
                           double cov_reg = 1e-6,
                           bool parallel = true,
                           bool verbose = true) {

   int T_all = X_all.n_rows;
   int D = X_all.n_cols;
   if (T_all < 10) stop("Too few observations");
   int initial_window = initial_multiplier * training_frequency;
   if (initial_window >= T_all) stop("initial_window >= available observations; reduce initial_multiplier or training_frequency");

   // build ends
   std::vector<int> ends;
   for (int e = initial_window; e <= T_all; e += training_frequency) ends.push_back(e);
   if (ends.back() != T_all) ends.push_back(T_all);
   int ntasks = (int)ends.size() - 1;
   if (ntasks <= 0) stop("No tasks created");

   arma::ivec signal(T_all); signal.fill(0);
   arma::ivec states(T_all); states.fill(0); // 0 means NA/unassigned
   List diagnostics(ntasks);

   // Pre-build valid combos (global, but combos will be filtered by nstates availability per combo)
   struct Combo { int nstates; int n_bull; int n_bear; };
   std::vector<Combo> global_combos;
   for (int i=0;i<nstates_cand.size();++i) {
     int ns = nstates_cand[i];
     for (int j=0;j<n_bull_cand.size();++j) {
       int nb = n_bull_cand[j];
       for (int k=0;k<n_bear_cand.size();++k) {
         int ne = n_bear_cand[k];
         if (ns <= 0) continue;
         if (nb < 0 || ne < 0) continue;
         if (nb + ne > ns) continue; // invalid
         global_combos.push_back({ns, nb, ne});
       }
     }
   }
   if (global_combos.empty()) stop("No valid hyperparameter combinations provided");

   for (int ti = 0; ti < ntasks; ++ti) {
     int train_end = ends[ti];
     int predict_end = ends[ti+1];
     mat X_train = X_all.rows(0, train_end-1);
     mat X_allwin = X_all.rows(0, predict_end-1);
     vec ret_vals = X_allwin.col(0); // returns used for scoring

     // Determine combos that are applicable: some combos require nstates <= something
     std::vector<Combo> combos;
     for (auto &c : global_combos) combos.push_back(c); // all combos valid here (we do not restrict by data size)
     if (combos.empty()) stop("No valid combos for this task (unexpected)");

     // Provide storage for best
     double best_score = -std::numeric_limits<double>::infinity();
     Combo best_combo = combos[0];
     HMMModel best_model;
     arma::uvec best_states_all;

     // iterate combos (optionally in parallel). We must be careful with thread-safety in train_hmm_em and Armadillo.
     size_t ncomb = combos.size();

     // Helper lambda to evaluate one combo (we will run either sequential or parallel)
     auto eval_combo = [&](size_t ci, double &out_score, HMMModel &out_model, arma::uvec &out_states_all, Combo &out_combo) {
       Combo c = combos[ci];
       out_combo = c;
       out_score = -std::numeric_limits<double>::infinity();

       // set a local seed to improve reproducibility (vary by task and combo)
       unsigned int local_seed = (unsigned int) (seed + ti * 10007 + (int)ci * 131);
       std::srand(local_seed);

       // train model (catch exceptions)
       HMMModel mdl;
       try {
         mdl = train_hmm_em(X_train, c.nstates, maxit, tol, cov_reg, false);
       } catch(...) {
         return; // leave out_score as -inf (skipped)
       }

       // decode on full window (train + OOS) to allow selection based on full-window state_means as in original
       arma::uvec states_candidate;
       try {
         states_candidate = viterbi_decode(mdl, X_allwin); // 1..K
       } catch(...) {
         return;
       }

       // compute state_means on full window
       vec state_means(c.nstates); state_means.fill(NA_REAL);
       vec state_sds(c.nstates); state_sds.fill(NA_REAL);
       for (int k=1;k<=c.nstates;k++) {
         uvec idx = find(states_candidate == (uword)k);
         if (idx.n_elem > 0) {
           vec vals(idx.n_elem);
           for (uword j=0;j<idx.n_elem;j++) vals(j) = ret_vals(idx(j));
           state_means(k-1) = mean(vals);
           state_sds(k-1) = (vals.n_elem > 1) ? stddev(vals) : 0.0;
         }
       }

       // create ordering by state_means descending (NA to end)
       std::vector<int> ord_states(c.nstates);
       std::iota(ord_states.begin(), ord_states.end(), 1);
       std::sort(ord_states.begin(), ord_states.end(), [&](int a, int b){
         double sa = state_means(a-1), sb = state_means(b-1);
         bool fa = std::isfinite(sa), fb = std::isfinite(sb);
         if (fa && fb) return sa > sb;
         if (fa && !fb) return true;
         if (!fa && fb) return false;
         return a < b;
       });

       // count occurrences in OOS part of window
       int from = train_end;
       int to = predict_end - 1;
       vec oos_counts(c.nstates); oos_counts.zeros();
       if (to >= from) {
         for (int t = from; t <= to; ++t) {
           int st = (int) states_candidate(t); // 1..K
           if (st >= 1 && st <= c.nstates) oos_counts(st-1) += 1;
         }
       }

       // choose bull_states and bear_states with preference for presence in OOS, as in original logic
       std::vector<int> bull_states;
       std::vector<int> bear_states;

       // bulls: top n_bull prioritizing OOS presence
       for (size_t ii=0; ii<ord_states.size() && (int)bull_states.size() < c.n_bull; ++ii) {
         int s = ord_states[ii];
         if (oos_counts(s-1) > 0) bull_states.push_back(s);
       }
       for (size_t ii=0; ii<ord_states.size() && (int)bull_states.size() < c.n_bull; ++ii) {
         int s = ord_states[ii];
         if (std::find(bull_states.begin(), bull_states.end(), s) == bull_states.end()) bull_states.push_back(s);
       }
       // bears: bottom n_bear prioritizing OOS presence
       for (int ii=(int)ord_states.size()-1; ii>=0 && (int)bear_states.size() < c.n_bear; --ii) {
         int s = ord_states[ii];
         if (oos_counts(s-1) > 0) bear_states.push_back(s);
       }
       for (int ii=(int)ord_states.size()-1; ii>=0 && (int)bear_states.size() < c.n_bear; --ii) {
         int s = ord_states[ii];
         if (std::find(bear_states.begin(), bear_states.end(), s) == bear_states.end()) bear_states.push_back(s);
       }

       // ensure disjoint, prefer bull
       std::vector<int> tmp;
       tmp.clear();
       for (int s: bull_states) if (std::find(bear_states.begin(), bear_states.end(), s) == bear_states.end()) tmp.push_back(s);
       bull_states = tmp;
       tmp.clear();
       for (int s: bear_states) if (std::find(bull_states.begin(), bull_states.end(), s) == bear_states.end()) tmp.push_back(s);
       bear_states = tmp;

       // Compute OOS metric: mean ret in OOS for bull minus mean ret for bear
       double mean_bull = 0.0, mean_bear = 0.0;
       int nbull_obs = 0, nbear_obs = 0;
       if (to >= from) {
         for (int t = from; t <= to; ++t) {
           int st = (int) states_candidate(t);
           if (std::find(bull_states.begin(), bull_states.end(), st) != bull_states.end()) {
             mean_bull += ret_vals(t);
             nbull_obs++;
           } else if (std::find(bear_states.begin(), bear_states.end(), st) != bear_states.end()) {
             mean_bear += ret_vals(t);
             nbear_obs++;
           }
         }
       }
       if (nbull_obs > 0) mean_bull /= nbull_obs; else mean_bull = NAN;
       if (nbear_obs > 0) mean_bear /= nbear_obs; else mean_bear = NAN;

       double score = -1e12;
       if (!std::isnan(mean_bull) && !std::isnan(mean_bear)) {
         score = mean_bull - mean_bear;
       } else {
         // penalize combos that do not produce both bull and bear in OOS
         score = -1e9 - ((nbull_obs + nbear_obs)); // more observations slightly better
       }

       // produce outputs
       out_score = score;
       out_model = mdl;
       out_states_all = states_candidate;
     }; // end lambda

     // containers for per-combo outputs if parallel
     std::vector<double> scores(ncomb, -std::numeric_limits<double>::infinity());
     std::vector<HMMModel> models(ncomb);
     std::vector<arma::uvec> states_all_vec(ncomb);
     std::vector<Combo> combos_out(ncomb);

     // Decide parallelism
#ifdef _OPENMP
     bool do_parallel = parallel && (ncomb > 1);
#else
     bool do_parallel = false;
#endif

     if (do_parallel) {
#ifdef _OPENMP
       // parallel loop over combos
#pragma omp parallel for schedule(dynamic)
       for (int ci = 0; ci < (int)ncomb; ++ci) {
         try {
           double sc; HMMModel mdl; arma::uvec stv; Combo outc;
           eval_combo(ci, sc, mdl, stv, outc);
           scores[ci] = sc;
           combos_out[ci] = outc;
           models[ci] = mdl;
           states_all_vec[ci] = stv;
         } catch(...) {
           // skip on errors
         }
       }
#endif
     } else {
       for (size_t ci = 0; ci < ncomb; ++ci) {
         try {
           double sc; HMMModel mdl; arma::uvec stv; Combo outc;
           eval_combo(ci, sc, mdl, stv, outc);
           scores[ci] = sc;
           combos_out[ci] = outc;
           models[ci] = mdl;
           states_all_vec[ci] = stv;
         } catch(...) {
           // skip
         }
       }
     }

     // pick best score
     for (size_t ci=0; ci<ncomb; ++ci) {
       if (!std::isfinite(scores[ci])) continue;
       if (scores[ci] > best_score) {
         best_score = scores[ci];
         best_combo = combos_out[ci];
         best_model = models[ci];
         best_states_all = states_all_vec[ci];
       }
     }

     // fallback: if no valid best (all -inf), choose a safe fallback: first combo trained successfully
     if (!std::isfinite(best_score)) {
       bool found = false;
       for (size_t ci=0; ci<ncomb; ++ci) {
         if (states_all_vec[ci].n_elem > 0) {
           best_score = scores[ci];
           best_combo = combos_out[ci];
           best_model = models[ci];
           best_states_all = states_all_vec[ci];
           found = true;
           break;
         }
       }
       if (!found) {
         // As last resort, train with minimal params (use first nstates_cand)
         int ns = nstates_cand[0];
         try {
           best_model = train_hmm_em(X_train, ns, maxit, tol, cov_reg, false);
           best_states_all = viterbi_decode(best_model, X_allwin);
           best_combo = {ns, 1, 1};
           if (verbose) Rcpp::Rcout << "Fallback model trained for task " << ti << "\n";
         } catch(...) {
           stop("All training attempts failed for task " + std::to_string(ti));
         }
       }
     }

     // Now with best_model and best_states_all, compute state_means, state_sds and selection sets to store diagnostics
     int Kbest = best_combo.nstates;
     vec state_means_best(Kbest); state_means_best.fill(NA_REAL);
     vec state_sds_best(Kbest); state_sds_best.fill(NA_REAL);
     for (int k=1;k<=Kbest;k++) {
       uvec idx = find(best_states_all == (uword)k);
       if (idx.n_elem > 0) {
         vec vals(idx.n_elem);
         for (uword j=0;j<idx.n_elem;j++) vals(j) = ret_vals(idx(j));
         state_means_best(k-1) = mean(vals);
         state_sds_best(k-1) = (vals.n_elem > 1) ? stddev(vals) : 0.0;
       }
     }

     // reproduction of selection used in eval_combo to determine bull/bear for final assignment
     std::vector<int> ord_states_best(Kbest);
     std::iota(ord_states_best.begin(), ord_states_best.end(), 1);
     std::sort(ord_states_best.begin(), ord_states_best.end(), [&](int a, int b){
       double sa = state_means_best(a-1), sb = state_means_best(b-1);
       bool fa = std::isfinite(sa), fb = std::isfinite(sb);
       if (fa && fb) return sa > sb;
       if (fa && !fb) return true;
       if (!fa && fb) return false;
       return a < b;
     });
     vec oos_counts_best(Kbest); oos_counts_best.zeros();
     int from = train_end;
     int to = predict_end - 1;
     if (to >= from) {
       for (int t = from; t <= to; ++t) {
         int st = (int) best_states_all(t);
         if (st >= 1 && st <= Kbest) oos_counts_best(st-1) += 1;
       }
     }
     std::vector<int> bull_states_best;
     std::vector<int> bear_states_best;
     for (size_t ii=0; ii<ord_states_best.size() && (int)bull_states_best.size() < best_combo.n_bull; ++ii) {
       int s = ord_states_best[ii];
       if (oos_counts_best(s-1) > 0) bull_states_best.push_back(s);
     }
     for (size_t ii=0; ii<ord_states_best.size() && (int)bull_states_best.size() < best_combo.n_bull; ++ii) {
       int s = ord_states_best[ii];
       if (std::find(bull_states_best.begin(), bull_states_best.end(), s) == bull_states_best.end()) bull_states_best.push_back(s);
     }
     for (int ii=(int)ord_states_best.size()-1; ii>=0 && (int)bear_states_best.size() < best_combo.n_bear; --ii) {
       int s = ord_states_best[ii];
       if (oos_counts_best(s-1) > 0) bear_states_best.push_back(s);
     }
     for (int ii=(int)ord_states_best.size()-1; ii>=0 && (int)bear_states_best.size() < best_combo.n_bear; --ii) {
       int s = ord_states_best[ii];
       if (std::find(bear_states_best.begin(), bear_states_best.end(), s) == bear_states_best.end()) bear_states_best.push_back(s);
     }
     // ensure disjoint
     {
       std::vector<int> tmpv;
       tmpv.clear();
       for (int s: bull_states_best) if (std::find(bear_states_best.begin(), bear_states_best.end(), s) == bear_states_best.end()) tmpv.push_back(s);
       bull_states_best = tmpv;
       tmpv.clear();
       for (int s: bear_states_best) if (std::find(bull_states_best.begin(), bull_states_best.end(), s) == bull_states_best.end()) tmpv.push_back(s);
       bear_states_best = tmpv;
     }

     // assign signals and states for OOS portion (train_end .. predict_end-1)
     int from_assign = train_end;
     int to_assign = predict_end - 1;
     if (to_assign >= from_assign) {
       for (int t = from_assign; t <= to_assign; ++t) {
         int st = (int) best_states_all(t); // 1..K
         if (std::find(bull_states_best.begin(), bull_states_best.end(), st) != bull_states_best.end()) signal(t) = 1;
         else if (std::find(bear_states_best.begin(), bear_states_best.end(), st) != bear_states_best.end()) signal(t) = -1;
         else signal(t) = 0;
         states(t) = st;
       }
     }

     // store diagnostics for this task
     List diag = List::create(
       Named("state_means") = state_means_best,
       Named("state_sds") = state_sds_best,
       Named("bull_states") = wrap(bull_states_best),
       Named("bear_states") = wrap(bear_states_best),
       Named("train_end") = train_end,
       Named("predict_end") = predict_end,
       Named("oos_counts") = oos_counts_best,
       Named("chosen_nstates") = best_combo.nstates,
       Named("chosen_n_bull") = best_combo.n_bull,
       Named("chosen_n_bear") = best_combo.n_bear,
       Named("chosen_score") = best_score
     );
     diagnostics[ti] = diag;

     if (verbose) {
       Rcpp::Rcout << "Task " << ti+1 << "/" << ntasks << " train_end=" << train_end << " predict_end=" << predict_end << "\n";
       Rcpp::Rcout << "  chosen (nstates,n_bull,n_bear)=(" << best_combo.nstates << "," << best_combo.n_bull << "," << best_combo.n_bear << ")"
                   << " score=" << best_score << "\n";
       Rcpp::Rcout << "  bull_states: ";
       for (int s: bull_states_best) Rcpp::Rcout << s << " ";
       Rcpp::Rcout << " bear_states: ";
       for (int s: bear_states_best) Rcpp::Rcout << s << " ";
       Rcpp::Rcout << " oos_counts: ";
       for (int k=0;k<Kbest;++k) Rcpp::Rcout << (int)oos_counts_best(k) << " ";
       Rcpp::Rcout << "\n";
     }

   } // end tasks

   return List::create(Named("signals") = signal,
                       Named("states") = states,
                       Named("diagnostics") = diagnostics);
 }
