data {
  int<lower=1> N;       // number of obs
  int<lower=1> K;       // number of covariates 
  int<lower=1> J;       // number of room types
  int<lower=1,upper=J> room_type[N]; // room type index
  matrix[N,K] X;        // matrix of covariates
  vector[N] y;          // log price 
}
parameters {
  vector[J] alpha;
  vector[K] beta;
  real mu;
  real<lower=0> sigma_a;  
  real<lower=0> sigma_y;
}

model {    
  // likelihood 
  vector[N] y_hat;
  for (i in 1:N){
    y_hat[i] = alpha[room_type[i]] + X[i]*beta;
  }
  y ~ normal(y_hat, sigma_y);
  
  // priors
  mu ~ normal(0,1);
  beta ~ normal(0,1);
  sigma_a ~ normal(0,1);
  sigma_y ~ normal(0,1);
  
  // hierarchical 
  alpha ~ normal(mu, sigma_a);
}

generated quantities {
  vector[N] log_lik;        // pointwise log-likelihood for LOO
  vector[N] log_price_rep;  // replications from posterior predictive dist

  for (n in 1:N) {
    real log_price = alpha[room_type[n]] + X[n]*beta;
    log_lik[n] = normal_lpdf(y[n] | log_price, sigma_y);
    log_price_rep[n] = normal_rng(log_price, sigma_y);
  }
}

