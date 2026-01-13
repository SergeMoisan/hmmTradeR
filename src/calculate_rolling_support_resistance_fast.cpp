// src/support_resistance_fast.cpp
// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <deque>

using namespace Rcpp;

// --- sliding-window helpers (unchanged, return full series windows)
static NumericVector sliding_window_max(const NumericVector& adjusted, int width) {
  const int n = adjusted.size();
  const int num_windows = n - width + 1;
  if (num_windows <= 0) return NumericVector(0);

  NumericVector window_max(num_windows);
  std::deque<int> dq;
  int out_idx = 0;

  for (int i = 0; i < n; ++i) {
    while (!dq.empty() && adjusted[i] >= adjusted[dq.back()]) dq.pop_back();
    dq.push_back(i);
    if (!dq.empty() && dq.front() <= i - width) dq.pop_front();
    if (i >= width - 1) {
      window_max[out_idx++] = adjusted[dq.front()];
    }
  }
  return window_max;
}

static NumericVector sliding_window_min(const NumericVector& adjusted, int width) {
  const int n = adjusted.size();
  const int num_windows = n - width + 1;
  if (num_windows <= 0) return NumericVector(0);

  NumericVector window_min(num_windows);
  std::deque<int> dq;
  int out_idx = 0;

  for (int i = 0; i < n; ++i) {
    while (!dq.empty() && adjusted[i] <= adjusted[dq.back()]) dq.pop_back();
    dq.push_back(i);
    if (!dq.empty() && dq.front() <= i - width) dq.pop_front();
    if (i >= width - 1) {
      window_min[out_idx++] = adjusted[dq.front()];
    }
  }
  return window_min;
}

//' Calculate rolling support and resistance (faster)
 //' @export
 // [[Rcpp::export]]
 DataFrame calculate_rolling_support_resistance_fast(CharacterVector dates,
                                                     NumericVector opens,
                                                     NumericVector highs,
                                                     NumericVector lows,
                                                     NumericVector closes,
                                                     NumericVector volumes,
                                                     int window_size = 100,
                                                     int width = 20) {
   const int n = dates.size();
   if ((int)opens.size() != n || (int)highs.size() != n || (int)lows.size() != n || (int)closes.size() != n || (int)volumes.size() != n) stop(\"All input vectors must have the same length\");
   if (window_size < 1) stop(\"window_size must be >= 1\");
   if (width < 1) stop(\"width must be >= 1\");

   const NumericVector adjusted = closes;
   const int middle = (width + 1) / 2;

   // Precompute full-series sliding minima & maxima for given width
   NumericVector window_min_full = sliding_window_min(adjusted, width); // length n-width+1
   NumericVector window_max_full = sliding_window_max(adjusted, width);

   // Outputs
   NumericVector supports(n, NumericVector::get_na());
   NumericVector resistances(n, NumericVector::get_na());

   // For each i (end of historical window)
   for (int i = window_size; i < n; ++i) {
     const int start_idx = i - window_size; // inclusive
     const int end_idx = i; // inclusive
     const int len = end_idx - start_idx + 1;
     if (len < width) continue;

     // windows s span [start_idx .. end_idx - width + 1]
     int s_last = end_idx - width + 1;
     int s_first = start_idx;
     // search backwards for last support (the most recent)
     bool found_support = false;
     for (int s = s_last; s >= s_first; --s) {
       int center = s + middle - 1; // global center index
       if (center < start_idx || center > end_idx) continue; // safety
       if (NumericVector::is_na(adjusted[center])) continue;
       double center_val = adjusted[center];
       double win_min = window_min_full[s]; // s is global index for window start
       if (center_val == win_min) {
         supports[i] = center_val;
         found_support = true;
         break;
       }
     }
     if (!found_support) {
       supports[i] = NumericVector::get_na();
     }

     // search backwards for last resistance
     bool found_res = false;
     for (int s = s_last; s >= s_first; --s) {
       int center = s + middle - 1;
       if (center < start_idx || center > end_idx) continue;
       if (NumericVector::is_na(adjusted[center])) continue;
       double center_val = adjusted[center];
       double win_max = window_max_full[s];
       if (center_val == win_max) {
         resistances[i] = center_val;
         found_res = true;
         break;
       }
     }
     if (!found_res) {
       resistances[i] = NumericVector::get_na();
     }
   }

   return DataFrame::create(_[\"Date\"] = dates,
                            _[\"Open\"] = opens,
                            _[\"High\"] = highs,
                            _[\"Low\"] = lows,
                            _[\"Close\"] = closes,
                            _[\"Volume\"] = volumes,
                            _[\"Support\"] = supports,
                            _[\"Resistance\"] = resistances);
 }
