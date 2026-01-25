test_that("forward_probs_live_cpp renvoie des probabilités valides", {

  # 1. Setup d'un scénario minimal (2 états, 1 dimension)
  # État 1: Bull (Moyenne haute), État 2: Bear (Moyenne basse)
  K <- 2
  D <- 1
  pi <- c(0.5, 0.5)
  A <- matrix(c(0.9, 0.1,  # Reste en Bull ou passe en Bear
                0.2, 0.8), # Passe en Bull ou reste en Bear
              nrow = K, byrow = TRUE)

  mu <- matrix(c(0.01, -0.01), nrow = K) # +1% vs -1%

  # Covariances simples (Sigma = 0.0001, donc Sigma_inv = 10000)
  # log(det(0.0001)) = -9.21034
  Sigma_inv_list <- list(matrix(10000), matrix(10000))
  logdetSigma <- c(-9.21034, -9.21034)

  # Données : Une série de retours très positifs (clairement Bull)
  X_recent <- matrix(c(0.012, 0.008, 0.015), ncol = D)

  # 2. Appel de la fonction
  probs <- forward_probs_live_cpp(
    X_recent = X_recent,
    pi = pi,
    A = A,
    mu = mu,
    Sigma_inv_list = Sigma_inv_list,
    logdetSigma = logdetSigma
  )

  # 3. Vérifications (Assertions)

  # Test A : La somme doit être strictement égale à 1
  expect_equal(sum(probs), 1, tolerance = 1e-10)

  # Test B : Avec des retours positifs, la probabilité de l'état 1 (Bull)
  # doit être supérieure à celle de l'état 2 (Bear)
  expect_gt(probs[1], probs[2])

  # Test C : Vérifier le type et la dimension
  expect_type(probs, "double")
  expect_length(probs, K)
})
