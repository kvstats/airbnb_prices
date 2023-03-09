data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of covariates
  vector[N] y;    // log price
  matrix[N,K] X;  // 
}

parameters {
  real<lower=0> sigma;
  vector[K] beta;
}

transformed parameters {
  vector[N] mu;
  mu = X * beta;
}

model {
  sigma ~ normal(0, 1);
  beta ~ normal(0, 1);
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] log_price_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real log_price = X[n] * beta;
    log_lik[n] = normal_lpdf(y[n] | log_price, sigma);
    log_price_rep[n] = normal_rng(log_price, sigma);
  }
}
