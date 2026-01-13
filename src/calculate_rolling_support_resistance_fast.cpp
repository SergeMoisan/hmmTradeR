// src/support_resistance_fast.cpp
// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <deque>

using namespace Rcpp;
//' @export
// [[Rcpp::export]]
DataFrame calculate_rolling_support_resistance_fast(CharacterVector dates,
                                                    NumericVector opens,
                                                    NumericVector highs,
                                                    NumericVector lows,
                                                    NumericVector closes,
                                                    NumericVector volumes,
                                                    NumericVector adjusted,
                                                    int window_size = 100,
                                                    int width = 20) {
  using Rcpp::NumericVector;
  using Rcpp::CharacterVector;
  using Rcpp::DataFrame;
  using std::deque;
  const int n = dates.size();
  if ((int)opens.size() != n || (int)highs.size() != n || (int)lows.size() != n || (int)closes.size() != n || (int)volumes.size() != n)
    Rcpp::stop("All input vectors must have the same length");
  if (window_size < 1) Rcpp::stop("window_size must be >= 1");
  if (width < 1) Rcpp::stop("width must be >= 1");
  if (width > n) Rcpp::stop("width cannot exceed series length");

  //const NumericVector adjusted = closes;
  const int middle = (width + 1) / 2;
  const double tol = 1e-12; // ou adaptez

  // sliding min/max over full series (length n - width + 1)
  auto sliding_min = [&](const NumericVector &x, int k) {
    int m = x.size();
    if (k > m) return NumericVector(0);
    NumericVector out(m - k + 1);
    deque<int> dq; // indices
    for (int i = 0; i < m; ++i) {
      // pop front if out of window
      while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
      // skip NA: if x[i] is NA, flush? here we treat NA as not comparable -> push index but handle NA when reading.
      while (!dq.empty()) {
        double xv = x[dq.back()];
        double xi = x[i];
        if (Rcpp::NumericVector::is_na(xi)) break;
        if (Rcpp::NumericVector::is_na(xv)) dq.pop_back();
        else {
          if (xi <= xv) dq.pop_back();
          else break;
        }
      }
      dq.push_back(i);
      if (i >= k - 1) {
        int start = i - k + 1;
        // ensure the front is valid
        while (!dq.empty() && dq.front() < start) dq.pop_front();
        if (dq.empty() || Rcpp::NumericVector::is_na(x[dq.front()])) out[start] = NumericVector::get_na(); else out[start] = x[dq.front()];
      }
    }
    return out;
  };

  auto sliding_max = [&](const NumericVector &x, int k) {
    int m = x.size();
    if (k > m) return NumericVector(0);
    NumericVector out(m - k + 1);
    deque<int> dq;
    for (int i = 0; i < m; ++i) {
      while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
      while (!dq.empty()) {
        double xv = x[dq.back()];
        double xi = x[i];
        if (Rcpp::NumericVector::is_na(xi)) break;
        if (Rcpp::NumericVector::is_na(xv)) dq.pop_back();
        else {
          if (xi >= xv) dq.pop_back();
          else break;
        }
      }
      dq.push_back(i);
      if (i >= k - 1) {
        int start = i - k + 1;
        while (!dq.empty() && dq.front() < start) dq.pop_front();
        if (dq.empty() || Rcpp::NumericVector::is_na(x[dq.front()])) out[start] = NumericVector::get_na(); else out[start] = x[dq.front()];
      }
    }
    return out;
  };

  NumericVector window_min_full = sliding_min(adjusted, width);
  NumericVector window_max_full = sliding_max(adjusted, width);

  NumericVector supports(n, NumericVector::get_na());
  NumericVector resistances(n, NumericVector::get_na());

  // i is the inclusive end index of the historical window of length window_size
  for (int i = window_size - 1; i < n; ++i) {
    int start_idx = i - window_size + 1; // inclusive
    int end_idx = i; // inclusive
    int len = end_idx - start_idx + 1;
    if (len < width) continue;

    int s_first = start_idx;
    int s_last = end_idx - width + 1;
    // constrain s range to valid indices for window_min_full (0 .. n-width)
    int max_s = n - width;
    if (s_first > max_s) {
      // no valid small windows inside full-series sliding results
      continue;
    }
    if (s_last > max_s) s_last = max_s;
    if (s_first < 0) s_first = 0;

    bool found_support = false;
    for (int s = s_last; s >= s_first; --s) {
      int center = s + middle - 1;
      if (center < start_idx || center > end_idx) continue;
      if (Rcpp::NumericVector::is_na(adjusted[center])) continue;
      double center_val = adjusted[center];
      double win_min = window_min_full[s];
      if (Rcpp::NumericVector::is_na(win_min)) continue;
      if (std::fabs(center_val - win_min) <= tol) {
        supports[i] = center_val;
        found_support = true;
        break;
      }
    }

    bool found_res = false;
    for (int s = s_last; s >= s_first; --s) {
      int center = s + middle - 1;
      if (center < start_idx || center > end_idx) continue;
      if (Rcpp::NumericVector::is_na(adjusted[center])) continue;
      double center_val = adjusted[center];
      double win_max = window_max_full[s];
      if (Rcpp::NumericVector::is_na(win_max)) continue;
      if (std::fabs(center_val - win_max) <= tol) {
        resistances[i] = center_val;
        found_res = true;
        break;
      }
    }
  }

  return DataFrame::create(Rcpp::Named("Date") = dates,
                           Rcpp::Named("Open") = opens,
                           Rcpp::Named("High") = highs,
                           Rcpp::Named("Low") = lows,
                           Rcpp::Named("Close") = closes,
                           Rcpp::Named("Volume") = volumes,
                           Rcpp::Named("Adjusted") = adjusted,
                           Rcpp::Named("Support") = supports,
                           Rcpp::Named("Resistance") = resistances);
}
