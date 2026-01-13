// src/support_resistance_fast.cpp
// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
using namespace Rcpp;

sr_fill_nas <- function(sr_data, first_col_fill = 0, cols_start = 1L) {
# 1) validations d'entrée
  if (!is.data.frame(sr_data) && !is.matrix(sr_data)) {
    stop("sr_data doit être un data.frame ou une matrix.")
  }
  if (!requireNamespace("zoo", quietly = TRUE)) {
    stop("Le package 'zoo' est nécessaire. Veuillez l'installer (install.packages('zoo')).")
  }
# sécuriser indices
  ncols <- ncol(sr_data)
    if (ncols == 0) return(sr_data)
      if (!is.numeric(cols_start) || length(cols_start) != 1 || cols_start < 1) {
        stop("cols_start doit être un entier >= 1.")
      }
      cols_start <- as.integer(cols_start)

# 2) traiter la première colonne demandée (par défaut colonne 1)
        first_idx <- cols_start
# travailler sur une copie pour conserver classes
        for (i in seq_len(ncols)) {
# si colonne est factor, convertir en character avant remplissage puis revenir si nécessaire
          if (is.factor(sr_data[, i])) {
            sr_data[, i] <- as.character(sr_data[, i])
          }
        }

# Remplir les leading NA de la colonne first_idx par first_col_fill (sans toucher aux autres NA)
        col_vec <- sr_data[, first_idx]
# on ne veut remplacer que les NA initiaux consécutifs au début
        if (any(is.na(col_vec))) {
          nas <- is.na(col_vec)
# position du premier non-NA
          first_non_na <- which(!nas)[1]
          if (is.na(first_non_na)) {
# toute la colonne est NA -> remplacer tout par first_col_fill
            col_vec[] <- first_col_fill
          } else if (first_non_na > 1) {
# remplacer les NA de 1:(first_non_na-1) par first_col_fill
            col_vec[1:(first_non_na - 1)] <- first_col_fill
          }
          sr_data[, first_idx] <- col_vec
        }

# 3) boucle à partir de la colonne suivante (cols_start + 1)
        if (cols_start < ncols) {
          for (i in seq(cols_start + 1, ncols)) {
# si la colonne est numérique / character / logical, na.locf fonctionne.
# pour sécurité, on travaille sur un vecteur : zoo::na.locf garde la classe (numeric/character)
            vec <- sr_data[, i]
# si toute la colonne est NA, na.locf la laisse NA ; on peut décider d'une stratégie :
# ici on applique na.locf sans na.rm = FALSE (conserver les leading NA si pas de valeur précédente)
            vec_filled <- zoo::na.locf(vec, na.rm = FALSE)
# si vous souhaitez remplacer encore les leading NA résultants par 0, on pourrait le faire,
# mais la demande initiale veut seulement remplir la 1ère colonne par 0 puis na.locf à partir de la 2ème.
            sr_data[, i] <- vec_filled
          }
        }

# 4) si des colonnes étaient factors initialement, on n'a pas reconverti (conserver as.character est souvent préférable)
# si vous souhaitez restaure les factors, il faudrait passer par un argument ou détecter rang initial.

        return(sr_data)
}
