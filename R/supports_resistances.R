#' Calcul rapide des supports et résistances roulants
#'
#' Calcule des niveaux de support et de résistance par fenêtre roulante à partir
#' de séries OHLC ajustées. Implémentation en C++ pour performance (Rcpp).
#'
#' @param dates Character vector de dates (ou identifiants de période).
#' @param opens Numeric vector des prix d'ouverture.
#' @param highs Numeric vector des prix hauts.
#' @param lows Numeric vector des prix bas.
#' @param closes Numeric vector des prix de clôture.
#' @param volumes Numeric vector des volumes.
#' @param adjusted Numeric vector des prix ajustés (utilisés pour les calculs).
#' @param window_size Entier >= 1 : taille de la fenêtre historique (nombre de barres).
#' @param width Entier >= 1 : largeur de la petite fenêtre utilisée pour détecter
#'   minima/maxima locaux (doit être <= longueur de la série).
#'
#' @return Un data.frame avec les colonnes :
#' \itemize{
#'   \item Date, Open, High, Low, Close, Volume, Adjusted (reprise des entrées)
#'   \item Support : valeur du support détecté à chaque indice (NA si aucun).
#'   \item Resistance : valeur de la résistance détectée à chaque indice (NA si aucun).
#' }
#'
#' @details
#' - La fonction est implémentée en C++ via Rcpp pour des performances élevées sur
#'   de longues séries. Les valeurs NA dans les entrées sont gérées : si la
#'   valeur centrale d'une petite fenêtre est NA elle est ignorée.
#' - Si width > length(series) la fonction stoppe.
#'
#' @examples
#' # (Exemple minimal — remplacer par des séries réelles)
#' dates <- as.character(Sys.Date() + 1:200)
#' x <- cumsum(rnorm(200))
#' df <- calculate_rolling_support_resistance_fast(dates,
#'                                                opens = x,
#'                                                highs = x + runif(200, 0, 1),
#'                                                lows = x - runif(200, 0, 1),
#'                                                closes = x,
#'                                                volumes = rep(1, 200),
#'                                                adjusted = x,
#'                                                window_size = 100,
#'                                                width = 21)
#' head(df)
#'
#' @seealso \code{\link[Rcpp]{Rcpp}}
#' @export

