#' sr_fill_nas: remplit les Na de la première par des 0 puis utilise LOCF.
#'
#' Cette fonction remplit les éventuels NA présents dans un jeu de données.
#' de supports et résistances en utilisant la méthode Last Observation Carried Forward (LOCF).
#' Cela permet de maintenir la continuité des niveaux critiques dans l'analyse technique.
#'
#' @param sr_data data.frame ou matrix numérique contenant les niveaux de supports et résistances,
#'                 dont les NA seront remplis colonne par colonne.
#'
#' @return Un objet du même type que \code{sr_data} avec les NA remplacés par la dernière valeur non manquante observée.
#' @export
#'
#' @importFrom zoo na.locf
#'
#' @examples
#' \dontrun{
#' library(zoo)
#' sr_data <- data.frame(
#'   Support = c(1.1, NA, NA, 1.4, 1.5),
#'   Resistance = c(2.1, 2.2, NA, NA, 2.5)
#' )
#' sr_fill_nas(sr_data)
#' }
sr_fill_nas <- function(sr_data) {
  # s'assure que le package 'zoo' est chargé pour na.locf
  if (!requireNamespace("zoo", quietly = TRUE)) {
    stop("Le package 'zoo' est nécessaire pour utiliser cette fonction. Veuillez l'installer.")
  }
  sr_data[1, is.na(sr_data[1,])] <- 0
  for (i in 2:ncol(sr_data)){
    sr_data[, i] <- zoo::na.locf(sr_data[, i], na.rm = FALSE)
  }
  return(sr_data)
}
