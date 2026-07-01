// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <omp.h>
using namespace Rcpp;


Function txtProgressBar("txtProgressBar");
Function setTxtProgressBar("setTxtProgressBar");
Function close("close");
Rcpp::Function cov_fun("cov");
Function splineDesign("splineDesign", Environment::namespace_env("splines"));
Function bs("bs", Environment::namespace_env("splines"));
Function splinefun("splinefun");

// Estrae un valore dalla lista; se assente o NULL usa il default
template <typename T>
T get_or_default(const Rcpp::List& lst, const std::string& name, T default_val) 
{
  if (lst.containsElementNamed(name.c_str())) {
    SEXP elem = lst[name];
    if (!Rf_isNull(elem)) {
      return Rcpp::as<T>(elem);
    }
  }
  return default_val;
}

// Estrae un Nullable<T> dalla lista; se assente o NULL resta NULL
template <typename T>
Rcpp::Nullable<T> get_nullable(const Rcpp::List& lst, const std::string& name) 
{
  if (lst.containsElementNamed(name.c_str())) {
    SEXP elem = lst[name];
    if (!Rf_isNull(elem)) {
      return Rcpp::Nullable<T>(elem);
    }
  }
  return R_NilValue;
}

// ==========================
// bs_cpp 
// ==========================

arma::mat bs_rcpp(const arma::vec& x,const arma::vec& knots,
                      int degree = 3,
                      bool intercept = false,
                      Nullable<NumericVector> Boundary_knots = R_NilValue) {
  // Input : x = points where to evaluate the B-spline basis
  //         knots = internal knots of the B-spline basis
  //         degree = degree of the B-spline basis  
  //         Boundary_knots = optional vector of length 2 specifying the boundary knots
  // Output: B-spline basis matrix evaluated at points x
  
  NumericMatrix B;
  
  if (Boundary_knots.isNull()) {
     B = bs(x,
                         _["knots"] = knots,
                         _["degree"] = degree,
                         _["intercept"] = intercept);
  } else {
     B =  bs(x,
                          _["knots"] = knots,
                          _["degree"] = degree,
                          _["intercept"] = intercept,
                          _["Boundary.knots"] = Boundary_knots);
  }
  
  return Rcpp::as<arma::mat>(B);
}


// ==========================
// b_splines_cpp
// ==========================

arma::vec b_splines_cpp(
    int n_knots,
    const arma::vec& time_vec,
    const arma::vec& coeff_vec,
    int degree,
    bool intercept,
    Nullable<NumericVector> boundary_knots = R_NilValue // nuovo argomento
)
  {
  // Input : n_knots = number of knots (including boundary knots)
  //         time_vec = vector of time points where to evaluate the B-splines   
  //         coeff_vec = optional vector of coefficients for linear combination
  //         cond_order = if true, use order 4 B-splines, else order 3
  //         boundary_knots = optional vector of length 2 specifying boundary knots 
  // Output: if coeff_vec is NULL, returns the B-spline basis matrix
  //         else returns the vector obtained by multiplying the basis matrix
  
  int order = intercept ? degree + 1 : degree;
  int n_internal = n_knots - order;
  
  arma::vec knots_internal(n_internal, arma::fill::zeros);
  
  for (int i = 0; i < n_internal; i++)
    knots_internal[i] = (i + 1.0) / (n_internal + 1.0);
  
  arma::mat B = bs_rcpp(time_vec,knots_internal,degree,intercept,boundary_knots);
  
  arma::vec out = B * coeff_vec;
  
  return out;
}

arma::vec interp_spline_cpp(const arma::vec& x,
                            const arma::vec& y,
                            int nout)
{
  
  double xmin = x.min();
  double xmax = x.max();
  
  arma::vec ind_out(nout);
  
  if (nout == 1) {
    ind_out[0] = xmin;
  } else {
    double step = (xmax - xmin) / (nout - 1);
    for (int i = 0; i < nout; i++) {
      ind_out[i] = xmin + i * step;
    }
  }
  
  Function spfit = splinefun(wrap(x), wrap(y));
  
  // spfit(ind_out)
  NumericVector res = spfit(wrap(ind_out));
  
  return as<arma::vec>(res);
  
}

Rcpp::List build_y_star(
   const Rcpp::List& h_p,
    const std::vector<arma::mat>& y,
    const std::vector<Rcpp::IntegerVector>& n_obs,
    const Rcpp::IntegerVector& n_per_group)
{
  
  int n_groups = h_p.size();
  Rcpp::List y_star(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    int n_i = n_per_group[g];
    Rcpp::List y_star_g(n_i);
    
    Rcpp::List h_p_g = h_p[g];
    arma::mat y_g    = y[g];
    Rcpp::IntegerVector n_obs_g = n_obs[g];
    
    for (int i = 0; i < n_i; i++) {
      
      int n_obs_gi = n_obs_g[i];
      
      arma::vec h_gi = Rcpp::as<arma::vec>(h_p_g[i]);
      
      // prendi solo i valori osservati (niente NaN)
      arma::vec y_gi = y_g.col(i).subvec(0, n_obs_gi - 1);
      
      // interpolazione spline
      arma::vec y_star_gi =
        interp_spline_cpp(h_gi, y_gi, y_gi.n_elem);
      
      y_star_g[i] = y_star_gi;
    }
    
    y_star[g] = y_star_g;
  }
  
  return y_star;
}

Rcpp::List build_y_star_single_group(
    const Rcpp::List& h_p,
    const arma::mat y,
    const Rcpp::IntegerVector& n_obs,
    int n_pat)
{

  Rcpp::List y_star(n_pat);
    
  for (int i = 0; i < n_pat; i++) {
      
    int n_obs_gi = n_obs[i];
    arma::vec h_gi = Rcpp::as<arma::vec>(h_p[i]);
  
    arma::vec y_gi = y.col(i).subvec(0, n_obs_gi - 1);
      
    arma::vec y_star_gi =  interp_spline_cpp(h_gi, y_gi, y_gi.n_elem);
      
    y_star[i] = y_star_gi;

  }
    
  return y_star;
}

void check_monotonicity(const arma::vec& phi, bool LAND)
{
    if (phi.n_elem < 2) return;

    for (arma::uword i = 1; i < phi.n_elem; ++i) {
        if (phi(i) <= phi(i-1)) {
            if(LAND){Rcpp::stop("phi_init_mat must have all strictly increasing rows");}
            else{Rcpp::stop("phi_init must be strictly increasing");}
        }
    }
}

void check_boundary(arma::vec phi, double lower, double upper,bool LAND)
{
  int q = phi.n_elem; 

  if(phi[0] != lower){
    if(LAND){Rcpp::stop("phi_init_landmark does not satisfy the landmark conditions");}
    else{Rcpp::stop("The first component of phi_init must be " + std::to_string(lower));} 
  }
  
  if(phi[q-1] != upper){

    if(LAND){Rcpp::stop("phi_init_landmark does not satisfy the landmark conditions");}
    else{ Rcpp::stop("The last component of phi_init must be " + std::to_string(upper));} 
  }

}

void check_phi_warping(arma::vec phi, double lower = 0, double upper = 1, bool LAND = false)
{
  check_monotonicity(phi,LAND); 

  check_boundary(phi,lower,upper,LAND); 
}


// ==========================
// Omega_P
// ==========================
arma::mat omega_P_cpp(int dim) 
{
  // Input: dimension of the matrix
  // Output: matrix of penalization obtained from the random walk of order 1
  
  if (dim <= 0) {
    Rcpp::stop("dim must be positive");
  }
  
  arma::mat K = arma::zeros(dim, dim);
  
  // diagonale principale
  for (int i = 0; i < dim - 1; i++) {
    K(i, i) = 2.0;
  }
  K(dim - 1, dim - 1) = 1.0;
  
  // sotto- e sopra-diagonale
  for (int i = 0; i < dim - 1; i++) {
    K(i + 1, i) = -1.0; // sotto
    K(i, i + 1) = -1.0; // sopra
  }
  
  return K;
}


// ==========================
// stack_matrix
// ==========================
std::vector<arma::mat> stack_Matrix_groups(const std::vector<std::vector<arma::mat>>& x_list, int ncol) 
{
  
  int G = x_list.size();
  if (G == 0) Rcpp::stop("x_list is empty");
  
  std::vector<arma::mat> out(G);
  
  for (int g = 0; g < G; g++) {
    
    std::vector<arma::mat> group_list = x_list[g];
    int n = group_list.size();
    if (n == 0) Rcpp::stop("group list is empty");
    
    arma::uword total_rows = 0;
    std::vector<arma::mat> mats(n);
    
    for (int i = 0; i < n; i++) {
      arma::mat Xi = group_list[i];
      
      if (Xi.n_cols != static_cast<arma::uword>(ncol))
        Rcpp::stop("all matrices must have the same number of columns");
      
      mats[i] = Xi;
      total_rows += Xi.n_rows;
    }
    
    // stack verticale per il gruppo g
    arma::mat Xg(total_rows, ncol);
    arma::uword row_offset = 0;
    
    for (int i = 0; i < n; i++) {
      Xg.rows(row_offset, row_offset + mats[i].n_rows - 1) = mats[i];
      row_offset += mats[i].n_rows;
    }
    
    out[g] = Xg;
  }
  
  return out;
}

arma::mat stack_single_group(const std::vector<arma::mat>& group_mats, int ncol, const Rcpp::IntegerVector& n_obs_g) 
{

    arma::uword total_rows = Rcpp::sum(n_obs_g);
    arma::mat Xg(total_rows, ncol);

    arma::uword row_offset = 0;
    for (size_t i = 0; i < group_mats.size(); i++) {
        const arma::mat& Xi = group_mats[i];  // riferimento per evitare copia
        Xg.rows(row_offset, row_offset + Xi.n_rows - 1) = Xi;
        row_offset += Xi.n_rows;
    }

    return Xg;
}

arma::mat stack_single_group_telesca(const std::vector<arma::mat>& group_mats, const arma::rowvec& a,
                                     int ncol, const Rcpp::IntegerVector& n_obs_g) 
{

    arma::uword total_rows = Rcpp::sum(n_obs_g);
    arma::mat Xg(total_rows, ncol);

    arma::uword row_offset = 0;
    for (size_t i = 0; i < group_mats.size(); i++) {
        const arma::mat& Xi = group_mats[i];
        double a_i = a(i);
        Xg.rows(row_offset, row_offset + Xi.n_rows - 1) = Xi * a_i;
        row_offset += Xi.n_rows;
    }

    return Xg;
}


// ==========================
// rho_clipping
// ==========================
double rho_clipping(double x,double x_min, double x_max) 
{
  // Input:  x = number to be clipped into the interval [x_min, x_max]
  //         x_min = lower bound of the interval
  //         x_max = upper bound of the interval
  // Output: clipped value of x
  
  if (x < x_min){ 
    return x_min;
  }
  if(x > x_max){ 
    return x_max; 
  }
  return x;
}

void check_rho_clipping(double x_min, double x_max)
{
  if (x_min >= x_max) {
    Rcpp::stop("Error: sd_min (" + std::to_string(x_min) + 
      ") cannot be greater or equal to sd_max (" + std::to_string(x_max) + ")");
    return; 
  }
}

arma::vec rmvnorm_1sample(
    arma::vec mu,
    arma::mat Sigma)
{
  // Input:  mu = mean vector
  //         Sigma = covariance matrix 
  // Output: vector with the sample from N(mu,Sigma)
  int p = mu.n_elem;
  
  arma::mat L = arma::chol(Sigma, "lower");
  
  arma::vec z = arma::randn<arma::vec>(p);
  
  arma::vec x = mu + L * z;
  
  return x;
}

double rnorm_1sample(double mu, double sigma) 
{
    // Input: mu = media
    //        sigma = deviazione standard
    // Output: un singolo valore campionato da N(mu, sigma^2)
    
    return mu + sigma * arma::randn<double>();
}

double rgamma_1sample(
    double shape,
    double rate) 
{
  // Input : shape = shape parameter of the gamma 
  //         rate  = rate parameter of the gamma 
  // Output: one samples from a gamma(shape,rate)
  
  double scale = 1.0 / rate;
  
  double out = R::rgamma(shape, scale);

  return out;
}

double rinvgamma_1sample(
    double shape,
    double rate)
{
  // Input : shape = shape parameter of the inv-gamma 
  //         rate  = rate parameter of the inv-gamma 
  // Output: one sample from a inv-gamma(shape,rate)
  
  double out = 1.0 / rgamma_1sample(shape, rate);
  return out; 
}

// ==========================
// update_beta_params
// ==========================

arma::mat compute_post_cov_beta(const arma::mat& Xg, const arma::mat& Omega, double lambda, double sigma_eps) 
{
    // Input : X = list of stucked Bm_beta matrices for each group g
  //         y_val = list of y vectors for each group g     
  //         Bm_gamma_group = list of lists of Bm matrices for each group g and subject i
  //         gamma_group = list of gamma matrices for each group g and subject i
  //         Omega = prior covariance matrix for beta
  //         beta_0 = prior mean vector for beta    
  //         lambda = current value of lambda
  //         sigma_eps = current value of sigma_eps
  //         iter = previuos iteration
  // Output: List with V_beta and m_beta for each group g
  
  arma::mat inv_sigma_beta = Omega / lambda;
  
  arma::mat inv_V_beta = inv_sigma_beta + (Xg.t() * Xg) / sigma_eps;
  arma::mat V_beta = arma::inv_sympd(inv_V_beta);
  
  return V_beta; 
}

arma::vec compute_post_mean_beta(const arma::mat& Xg, const arma::vec& yg, const arma::mat& Omega,
                                 double lambda, double sigma_eps,
                                 const arma::vec& beta_0, const arma::vec& c_gamma, const arma::mat& V_beta )
{
  arma::mat inv_sigma_beta = Omega / lambda;
  arma::vec m_beta = V_beta * ( Xg.t() * ((yg - c_gamma)/sigma_eps) + inv_sigma_beta * beta_0 );
  return m_beta;
}

// ==========================
// update_gamma_params
// ==========================
arma::mat compute_post_cov_gamma(const arma::mat& Bm_beta_gi, const arma::vec& beta_g, const arma::mat& Bm_gamma_gi,
                                 const arma::vec& y_gi,  
                                 double sigma_gamma, double sigma_eps )
{
  int k = Bm_gamma_gi.n_cols;
  
  arma::mat inv_sigma_gamma_matrix = arma::eye(k, k) / sigma_gamma;
  arma::mat prod_X_tilde =(Bm_gamma_gi.t() * Bm_gamma_gi) / sigma_eps;
  
  arma::mat inv_V_gamma = prod_X_tilde + inv_sigma_gamma_matrix;
  arma::mat V_gamma = arma::inv_sympd(inv_V_gamma);
  
  return V_gamma; 
}

arma::mat compute_post_mean_gamma(const arma::mat& Bm_beta_gi, const arma::vec& beta_g, const arma::mat& Bm_gamma_gi,
                                  const arma::vec& y_gi,  
                                  double sigma_gamma, double sigma_eps, const arma::vec& gamma_0, const arma::mat& V_gamma)
{
  
  int k = gamma_0.n_elem;
  
  arma::mat inv_sigma_gamma_matrix = arma::eye(k, k) / sigma_gamma;
  
  arma::vec c_tilde = Bm_beta_gi * beta_g;
  
  arma::vec diff = y_gi - c_tilde;
  arma::vec prod_term = Bm_gamma_gi.t() * diff;
  
  arma::vec m_gamma = V_gamma * ( (1.0 / sigma_eps)  * prod_term + inv_sigma_gamma_matrix * gamma_0 );
  
  return m_gamma; 
}

arma::mat compute_post_cov_a_c(int n_obs_gi, const arma::mat& Bm_beta_gi, const arma::vec& beta_g, 
                               const arma::mat& inv_Sigma_a_c, double sigma_eps)
{
  
  arma::vec beta_term = Bm_beta_gi * beta_g;
  arma::vec vec_ones(n_obs_gi, arma::fill::ones);

  arma::mat W = arma::join_rows(vec_ones, beta_term);

  arma::mat mat1 = W.t() * W / sigma_eps; 
  
  arma::mat inv_V_ac = mat1 + inv_Sigma_a_c;
  
  return arma::inv_sympd(inv_V_ac);
}

arma::vec compute_post_mean_a_c(int n_obs_gi, const arma::mat& Bm_beta_gi, const arma::vec& beta_g, 
                                const arma::mat& inv_Sigma_a_c, double sigma_eps, 
                                const arma::vec& y_gi, const arma::vec& vec_c0_a0, const arma::mat& V_ac)
{
  arma::vec beta_term = Bm_beta_gi * beta_g;
  arma::vec vec_ones(n_obs_gi, arma::fill::ones);

  arma::mat W = arma::join_rows(vec_ones, beta_term);

  arma::vec m_ac = V_ac * ( W.t() * y_gi / sigma_eps + inv_Sigma_a_c * vec_c0_a0 );
  
  return m_ac;
}

// ==========================
// evaluate_density_adaptive_MH
// ==========================
double evaluate_density_gardella_gamma_adaptive_MH(
    const arma::vec& x,
    const arma::vec& beta,
    const arma::vec& gamma_i,
    double sigma_eps,
    const arma::vec& a_vec,
    double b,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta, int degree_gamma,
    bool intercept_beta = false,
    bool intercept_gamma = false,
    Nullable<NumericVector> boundary_beta = R_NilValue,  Nullable<NumericVector> boundary_gamma = R_NilValue)
{
  // Input :  x = vector of parameters (log-scale for the csi vector)
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: log-density value of the full conditional of the csi vector evaluated at x
  
  int n = x.size();
  
  // 1. exp(x)
  arma::vec exp_x = arma::exp(x);
  exp_x[0] = 0.0;

  double sum_exp = arma::accu(exp_x);
  
  arma::vec phi_exp = arma::cumsum(exp_x) / sum_exp;
  
  arma::vec warped_t = Bh * phi_exp;
  
  arma::vec prod1 = b_splines_cpp(beta.size(), warped_t, beta,degree_beta, intercept_beta,boundary_beta);
  arma::vec prod2 = b_splines_cpp(gamma_i.size(), warped_t, gamma_i,degree_gamma, intercept_gamma,boundary_gamma);
  
  arma::vec add1 = y_i - (prod1 + prod2);
  
  double term1 = arma::dot(add1, add1) / sigma_eps;
  
  arma::vec log_densities = (a_vec % log(exp_x.subvec(1, n-1))) - exp_x.subvec(1, n-1) / b - lgamma(a_vec); 
  double term2 = std::exp(arma::sum(log_densities));
  
  arma::vec exp_sub = exp_x.subvec(1, n-1); 
  double term3 = arma::prod(exp_sub); 
  
  double log_density = -0.5 * term1 + std::log(term2) + std::log(term3);
  return log_density;
}

double evaluate_density_gardella_gaussian_MH(
    const arma::vec& phi,
    const arma::vec& beta,
    const arma::vec& gamma_i,
    double sigma_eps,
    const arma::vec& phi_0,
    const arma::mat& Omega_phi,
    double sigma_phi,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta, int degree_gamma,
    bool intercept_beta = false,
    bool intercept_gamma = false,
    Nullable<NumericVector> boundary_beta = R_NilValue,  Nullable<NumericVector> boundary_gamma = R_NilValue) 
{
  // Input :  x = vector of parameters (log-scale for the csi vector)
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: log-density value of the full conditional of the csi vector evaluated at x
  
  arma::vec warped_t = Bh * phi;
  
  arma::vec prod1 = b_splines_cpp(beta.size(), warped_t, beta,degree_beta, intercept_beta,boundary_beta);
  arma::vec prod2 = b_splines_cpp(gamma_i.size(), warped_t, gamma_i,degree_gamma, intercept_gamma,boundary_gamma);
  
  arma::vec add1 = y_i - (prod1 + prod2);
  
  double term1 = arma::dot(add1, add1) / sigma_eps;
  
  
  
  arma::vec diff = phi - phi_0;
  double term2 = arma::as_scalar(diff.t() * Omega_phi * diff) / sigma_phi;
  
  double log_density = -0.5 * (term1 + term2);
  return log_density;
}

double evaluate_density_telesca_gaussian_MH(
    const arma::vec& phi,
    const arma::vec& beta,
    double a_i,
    double c_i,
    double sigma_eps,
    const arma::vec& phi_0,
    const arma::mat& Omega_phi,
    double sigma_phi,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta,
    bool intercept_beta = false,
    Nullable<NumericVector> boundary_beta = R_NilValue)
{
  // Input :  x = vector of parameters (log-scale for the csi vector)
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: log-density value of the full conditional of the csi vector evaluated at x
  
  arma::vec warped_t = Bh * phi;
  
  arma::vec prod1 = b_splines_cpp(beta.size(), warped_t, beta,degree_beta, intercept_beta,boundary_beta) * a_i;
  arma::vec prod2 = arma::vec(warped_t.n_elem).fill(c_i);
  arma::vec add1 = y_i - (prod1 + prod2);
  
  double term1 = arma::dot(add1, add1) / sigma_eps;
  
  arma::vec diff = phi - phi_0;
  double term2 = arma::as_scalar(diff.t() * Omega_phi * diff) / sigma_phi;
  
  double log_density = -0.5 * (term1 + term2);
  return log_density;
}

double evaluate_density_telesca_gamma_adaptive_MH(
    const arma::vec& x,
    const arma::vec& beta,
    double a_i,
    double c_i,
    double sigma_eps,
    const arma::vec& a_vec, 
    double b, 
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta,
    bool intercept_beta = false,
    Nullable<NumericVector> boundary_beta = R_NilValue)
{
  // Input :  x = vector of parameters (log-scale for the csi vector)
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: log-density value of the full conditional of the csi vector evaluated at x

  int n = x.size();
  
  // 1. exp(x)
  arma::vec exp_x = arma::exp(x);
  exp_x[0] = 0.0;

  double sum_exp = arma::accu(exp_x);
  
  arma::vec phi_exp = arma::cumsum(exp_x) / sum_exp;
  
  arma::vec warped_t = Bh * phi_exp;
  
  arma::vec prod1 = b_splines_cpp(beta.size(), warped_t, beta,degree_beta, intercept_beta,boundary_beta) * a_i;
  arma::vec prod2 = arma::vec(warped_t.n_elem).fill(c_i);
  arma::vec add1 = y_i - (prod1 + prod2);
  
  double term1 = arma::dot(add1, add1) / sigma_eps;
  
  arma::vec log_densities = (a_vec % log(exp_x.subvec(1, n-1))) - exp_x.subvec(1, n-1) / b - lgamma(a_vec); 
  double term2 = std::exp(arma::sum(log_densities));
  
  arma::vec exp_sub = exp_x.subvec(1, n-1); 
  double term3 = arma::prod(exp_sub); 
  
  double log_density = -0.5 * term1 + std::log(term2) + std::log(term3);
  return log_density;
}

// ==========================
// adaptive_MH
// ==========================

void MH_gardella_gamma_adapt(
    arma::vec& x_new,  
    double& alpha,
    int& accettato,
    const arma::vec& x_old,
    const arma::mat& zeta_old,   // covarianza
    const arma::vec& beta,
    const arma::vec& gamma_i,
    double sigma_eps,
    const arma::vec& a_vec,
    double b,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta, int degree_gamma,
    bool intercept_beta = false,
    bool intercept_gamma = false,
    Nullable<NumericVector> boundary_beta = R_NilValue,  Nullable<NumericVector> boundary_gamma = R_NilValue)
{
  // Input : x_old = vector representing the current state of the chain (log-scale of csi)
  //         zeta_old = matrix of adaptive covariance
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: list with new state, acceptance indicator, and acceptance probability
  
  int p = zeta_old.n_rows;
  
  arma::mat Sigma = arma::reshape(zeta_old, p, p);
  
  arma::vec mean = arma::zeros(p);
  arma::vec increment = arma::mvnrnd(mean, Sigma);
  
  // 2. x_candidate = x_old + c(0, increment)
  int n = x_old.n_elem;
  arma::vec x_candidate = x_old;      
  x_candidate.subvec(1, n-1) += increment; 
  
  // 3. log-densità
  double log_dens_old = evaluate_density_gardella_gamma_adaptive_MH(x_old, beta, gamma_i, sigma_eps, a_vec, b, Bh, y_i,
                                                                    degree_beta,degree_gamma,intercept_beta, intercept_gamma,
                                                                    boundary_beta,boundary_gamma);
  double log_dens_cand = evaluate_density_gardella_gamma_adaptive_MH(x_candidate, beta, gamma_i, sigma_eps, a_vec, b, Bh, y_i,
                                                                     degree_beta,degree_gamma,intercept_beta, intercept_gamma,
                                                                     boundary_beta,boundary_gamma);
  double quoziente = log_dens_cand - log_dens_old;
  
  // 4. accettazione
  double alpha_log = std::min(0.0, quoziente);
  double u = R::runif(0.0, 1.0);

  alpha = std::exp(alpha_log);
  
  if (u < alpha) { // accept
    accettato = 1;
    x_new = x_candidate;
  } else { // reject
    accettato = 0;
    x_new = x_old;
  }
  
}

void MH_gardella_gaussian(
    arma::vec& phi_new,  
    int& accettato,
    int component,
    const arma::vec& beta,
    const arma::vec& gamma_i,
    double sigma_eps,
    const arma::vec& phi_0,
    const arma::mat& Omega_phi,
    double sigma_phi,
    double coeff_var_phi,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta, int degree_gamma,
    bool intercept_beta = false,
    bool intercept_gamma = false,
    Nullable<NumericVector> boundary_beta = R_NilValue,  Nullable<NumericVector> boundary_gamma = R_NilValue)
{
  // Input : x_old = vector representing the current state of the chain (log-scale of csi)
  //         zeta_old = matrix of adaptive covariance
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: list with new state, acceptance indicator, and acceptance probability

  double lower = std::max(phi_new[component] - coeff_var_phi, phi_new[component - 1]);
  double upper = std::min(phi_new[component] + coeff_var_phi, phi_new[component + 1]);

  double phi_new_component = R::runif(lower, upper);
  arma::vec phi_candidate = phi_new;
  phi_candidate[component] = phi_new_component;
  
  // 3. log-densità
  double log_dens_old = evaluate_density_gardella_gaussian_MH(phi_new, beta, gamma_i, sigma_eps, phi_0,Omega_phi, sigma_phi,
                                                              Bh, y_i,degree_beta,degree_gamma,intercept_beta, intercept_gamma,
                                                              boundary_beta,boundary_gamma);
  double log_dens_cand = evaluate_density_gardella_gaussian_MH(phi_candidate, beta, gamma_i, sigma_eps, phi_0,Omega_phi, sigma_phi,
                                                              Bh, y_i,degree_beta,degree_gamma,intercept_beta, intercept_gamma,
                                                              boundary_beta,boundary_gamma);
  
  double quoziente = log_dens_cand - log_dens_old;
  
  // 4. accettazione
  double alpha_log = std::min(0.0, quoziente);
  double u = R::runif(0.0, 1.0);
  
  if (u < std::exp(alpha_log)) { // accept
    accettato = 1;
    phi_new[component] = phi_candidate[component];
  } else { // reject
    accettato = 0;
  }
  
}

void MH_telesca_gaussian(
    arma::vec& phi_new,  
    int& accettato,
    int component,
    const arma::vec& beta,
    double a_i, 
    double c_i,
    double sigma_eps,
    const arma::vec& phi_0,
    const arma::mat& Omega_phi,
    double sigma_phi,
    double coeff_var_phi,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta,
    bool intercept_beta = false,
    Nullable<NumericVector> boundary_beta = R_NilValue)
{
  // Input : x_old = vector representing the current state of the chain (log-scale of csi)
  //         zeta_old = matrix of adaptive covariance
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: list with new state, acceptance indicator, and acceptance probability

  double lower = std::max(phi_new[component] - coeff_var_phi, phi_new[component - 1]);
  double upper = std::min(phi_new[component] + coeff_var_phi, phi_new[component + 1]);

  double phi_new_component = R::runif(lower, upper);
  arma::vec phi_candidate = phi_new;
  phi_candidate[component] = phi_new_component;
  
  // 3. log-densità
  double log_dens_old = evaluate_density_telesca_gaussian_MH(phi_new, beta, a_i, c_i, sigma_eps, phi_0,Omega_phi, sigma_phi,
                                                              Bh, y_i,degree_beta,intercept_beta,boundary_beta);
  double log_dens_cand = evaluate_density_telesca_gaussian_MH(phi_candidate, beta, a_i, c_i, sigma_eps, phi_0,Omega_phi, sigma_phi,
                                                              Bh, y_i,degree_beta,intercept_beta,boundary_beta);
  
  double quoziente = log_dens_cand - log_dens_old;
  
  // 4. accettazione
  double alpha_log = std::min(0.0, quoziente);
  double u = R::runif(0.0, 1.0);
  
  if (u < std::exp(alpha_log)) { // accept
    accettato = 1;
    phi_new[component] = phi_candidate[component];
  } else { // reject
    accettato = 0;
  }
  
}

void MH_telesca_gamma_adapt(
    arma::vec& x_new,  
    double& alpha,
    int& accettato,
    const arma::vec& x_old,
    const arma::mat& zeta_old,
    const arma::vec& beta,
    double a_i, 
    double c_i,
    double sigma_eps,
    const arma::vec& a_vec, 
    double b,
    const arma::mat& Bh,
    const arma::vec& y_i,
    int degree_beta,
    bool intercept_beta = false,
    Nullable<NumericVector> boundary_beta = R_NilValue)
{
  // Input : x_old = vector representing the current state of the chain (log-scale of csi)
  //         zeta_old = matrix of adaptive covariance
  //         beta, gamma_i = vectors of spline coefficients
  //         sigma_eps = scalar of noise variance
  //         a_vec = vector of shape parameters for the csi prior
  //         b = scale parameter for the csi prior
  //         Bh = basis matrix for warping functions
  //         y_i = observed data vector
  //         cond_order_beta, cond_order_gamma = logicals for conditional order splines 
  // Output: list with new state, acceptance indicator, and acceptance probability

  int p = zeta_old.n_rows;
  
  arma::mat Sigma = arma::reshape(zeta_old, p, p);
  
  arma::vec mean = arma::zeros(p);
  arma::vec increment = arma::mvnrnd(mean, Sigma);
  
  // 2. x_candidate = x_old + c(0, increment)
  int n = x_old.n_elem;
  arma::vec x_candidate = x_old;      
  x_candidate.subvec(1, n-1) += increment; 
  
  // 3. log-densità
  double log_dens_old = evaluate_density_telesca_gamma_adaptive_MH(x_new, beta, a_i, c_i, sigma_eps, a_vec,b,
                                                                   Bh, y_i,degree_beta,intercept_beta,boundary_beta);
  double log_dens_cand = evaluate_density_telesca_gamma_adaptive_MH(x_candidate, beta, a_i, c_i, sigma_eps,a_vec,b,
                                                                    Bh, y_i,degree_beta,intercept_beta,boundary_beta);
  
  double quoziente = log_dens_cand - log_dens_old;
  
  // 4. accettazione
  double alpha_log = std::min(0.0, quoziente);
  double u = R::runif(0.0, 1.0);
  
  alpha = std::exp(alpha_log); 
  
  if (u < alpha) { // accept
    accettato = 1;
    x_new = x_candidate;
  } else { // reject
    accettato = 0;
    x_new = x_old;
  }
  
}

Rcpp::List extract_y_info(const std::vector<arma::mat>& y) 
{
  
  int n_groups = y.size();
  arma::vec n_per_group(n_groups, arma::fill::zeros);
  arma::vec n_obs_tot_per_group(n_groups, arma::fill::zeros);
  int n_patients = 0;
  
  std::vector<arma::vec> n_obs(n_groups);
  std::vector<arma::vec> y_val(n_groups);
  
  int n_obs_max = 0;
  
  for (int g = 0; g < n_groups; g++) {
    
    const arma::mat& yg = y[g];
    int N_g = yg.n_cols;
    int n_rows = yg.n_rows;
    
    if (n_rows > n_obs_max) n_obs_max = n_rows;
    
    n_per_group[g] = N_g;
    n_patients += N_g;
    
    arma::ivec n_obs_g(N_g, arma::fill::zeros);
    int total_obs_group = 0;
    
    // numero osservazioni per individuo
    for (int i = 0; i < N_g; i++) {
      int count = 0;
      while (count < n_rows && !std::isnan(yg(count, i))) {
        count++;
      }
      n_obs_g[i] = count;
      total_obs_group += count;
    }
    
    n_obs[g] = arma::conv_to<arma::vec>::from(n_obs_g);
    n_obs_tot_per_group[g] = total_obs_group;
    
    // y_val concatenato
    arma::vec y_val_g(total_obs_group);
    int idx = 0;
    for (int i = 0; i < N_g; i++) {
      for (int t = 0; t < n_obs_g[i]; t++) {
        y_val_g[idx++] = yg(t, i);
      }
    }
    y_val[g] = y_val_g;
  }
  
  return Rcpp::List::create(
    _["n_groups"] = n_groups,
    _["n_patients"] = n_patients,
    _["n_per_group"] = n_per_group,
    _["n_obs_max"] = n_obs_max,
    _["n_obs"] = n_obs,
    _["n_obs_tot_per_group"] = n_obs_tot_per_group,
    _["y_val"] = y_val
  );
}

std::vector<arma::mat> pad_list_of_lists(const List & cleaned_list, const std::vector<Rcpp::IntegerVector> & n_obs, 
                                         int n_obs_max) 
{
  int G = cleaned_list.size();
  std::vector<arma::mat> result(G);
  
  for(int g = 0; g < G; g++) {
    List group = cleaned_list[g];        // lista interna g
    IntegerVector n_obs_g = n_obs[g];    // numero righe valide
    int n_pg = group.size();
    
    int n_col = 0;
    if(n_pg > 0) {
      arma::mat first_mat = Rcpp::as<arma::mat>(group[0]);
      n_col = first_mat.n_cols;
    }
    
    // Creo matrice finale: n_obs_max righe, n_pg * n_col colonne
    arma::mat mat_final(n_obs_max, n_pg * n_col);
    mat_final.fill(arma::datum::nan);  // riempio tutto di NaN
    
    for(int i = 0; i < n_pg; i++) {
      arma::mat mat_clean = Rcpp::as<arma::mat>(group[i]);
      int n_valid = n_obs_g[i];
      
      // numero di colonne di questa sottolista
      int n_c = mat_clean.n_cols;
      
      if(n_valid > 0) {
        mat_final(arma::span(0, n_valid - 1), arma::span(i * n_col, i * n_col + n_c - 1)) = mat_clean;
      }
    }
    
    result[g] = mat_final;
  }
  
  return result;
}

arma::mat pad_list_of_lists_single_group(const Rcpp::List & cleaned_list,
                                         const Rcpp::IntegerVector & n_obs,
                                         int n_obs_max)
{
  int n_pg = cleaned_list.size();   // numero di soggetti nel gruppo
  
  int n_col = 0;
  if(n_pg > 0) {
    arma::mat first_mat = Rcpp::as<arma::mat>(cleaned_list[0]);
    n_col = first_mat.n_cols;
  }
  
  // matrice finale: n_obs_max righe, n_pg * n_col colonne
  arma::mat mat_final(n_obs_max, n_pg * n_col);
  mat_final.fill(arma::datum::nan);
  
  for(int i = 0; i < n_pg; i++) {
    arma::mat mat_clean = Rcpp::as<arma::mat>(cleaned_list[i]);
    int n_valid = n_obs[i];
    int n_c = mat_clean.n_cols;
    
    if(n_valid > 0) {
      mat_final(
        arma::span(0, n_valid - 1),
        arma::span(i * n_col, i * n_col + n_c - 1)
      ) = mat_clean;
    }
  }
  
  return mat_final;
}

void check_params_y(const List& info_y,Nullable<int> n_groups,Nullable<IntegerVector> n_per_group)
{
  // Input : info_y = list of information extracted from y
  //         n_groups,n_per_group = information passed by the user
  // Output: checks if the information match, prints errors if not 
  
  // valori "veri" estratti da y
  int n_groups_y = info_y["n_groups"];
  IntegerVector n_per_group_y = info_y["n_per_group"];
  
  /* -----------------------------
   controllo su n_groups
   ----------------------------- */
  if (n_groups.isNotNull()) {
    int n_groups_user = as<int>(n_groups);
    
    if (n_groups_user != n_groups_y) {
      std::string msg =
        "Error: n_groups passed by the user (" +
        std::to_string(n_groups_user) +
        ") does not match the number of groups inferred from y (" +
        std::to_string(n_groups_y) + ").";
      
      stop(msg);
    }
  }
  
  /* -----------------------------
   controllo su n_per_group
   ----------------------------- */
  if (n_per_group.isNotNull()) {
    IntegerVector n_per_group_user = as<IntegerVector>(n_per_group);
    
    if (n_per_group_user.size() != n_per_group_y.size()) {
      
      std::string msg =
        "Error: length of n_per_group passed by the user (" +
        std::to_string(n_per_group_user.size()) +
        ") does not match the number of groups inferred from y (" +
        std::to_string(n_per_group_y.size()) +
        ").";
      
      stop(msg);
    }
    
    for (int g = 0; g < n_per_group_y.size(); g++) {
      if (n_per_group_user[g] != n_per_group_y[g]) {
        std::string msg =
          "Error: n_per_group[" +
          std::to_string(g + 1) +
          "] passed by the user (" +
          std::to_string(n_per_group_user[g]) +
          ") does not match the number of individuals in group " +
          std::to_string(g + 1) +
          " inferred from y (" +
          std::to_string(n_per_group_y[g]) +
          ").";
        
        stop(msg);
        
      }
    }
  }
}

List compute_spline_order(int degree_type, bool intercept,std::string param) 
{
  // Input : degree_type = integer representing the degree of the splines
  //         intercept = bool (T= include the intercept, F = discard the intercept)
  //         param = string representing the current parameter
  // Output: list containing the order of the spline and the intercept
  int order_spline;
  
  bool intercept_forzato = false;
  
  if (degree_type == 1) {
    order_spline = 2;
    if (!intercept) {
      intercept = true; 
      intercept_forzato = true; 
    }
  } else if (degree_type > 1) {
    order_spline = intercept ? degree_type + 1 : degree_type;
  } else {
    Rcpp::stop("degree must be >= 1");
  }
  
  if (intercept_forzato) {
    Rcpp::Rcout << "Warning: intercept for parameter '" << param 
                << "' forced to TRUE because degree_" <<param
                <<" = 1, even though user set FALSE." << std::endl;
  }
  
  return List::create(
    _["order_spline"] = order_spline,
    _["intercept"] = intercept
  );
}

arma::vec build_knots(int n_knots) 
{
  
  if (n_knots < 1) {
    Rcpp::stop("n_knots must be >= 1");
  }
  
  int n_total = n_knots + 2; // numero totale di punti compresi 0 e 1
  arma::vec knots(n_knots); // risultato finale
  
  for (int i = 0; i < n_knots; i++) {
    // scala tra 0 e 1, saltando il primo e l'ultimo
    knots[i] = static_cast<double>(i + 1) / (n_total - 1);
  }
  
  return knots;
}

std::vector<arma::mat> build_time_matrix(const std::vector<Rcpp::IntegerVector> & n_obs, int n_obs_max)
{
  // Input : n_obs = list of group vectors whose elements are the number of observations of the individual 
  //         n_obs_max = the maximum number of observations 
  // Output: list of group matrices whose columns are the time vectors for the individuals
  
  int n_groups = n_obs.size();
  std::vector<arma::mat> time_group(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    Rcpp::IntegerVector n_obs_g = n_obs[g]; 
    int N_g = n_obs_g.size();
    
    arma::mat time_mat(n_obs_max, N_g);
    
    std::fill(time_mat.begin(), time_mat.end(), NA_REAL);
    
    for (int i = 0; i < N_g; i++) {
      for (int t = 0; t < n_obs_g[i]; t++) {
        time_mat(t, i) = static_cast<double>(t) / (n_obs_g[i] - 1);
      }
    }
    
    time_group[g] = time_mat;
  }
  
  return time_group;
}

std::vector<std::vector<arma::mat>> build_Bm(
    const std::vector<arma::mat> & time_group, 
    int n_groups,
    const std::vector<Rcpp::IntegerVector> & n_obs,           
    const arma::vec & knots,
    int degree,
    bool intercept,
    Nullable<NumericVector> boundary) 
{
  // Input : 
  
  std::vector<std::vector<arma::mat>> Bm(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    arma::mat Tg = time_group[g];
    IntegerVector n_obs_g = n_obs[g];
    int N_g = Tg.n_cols;
    
    std::vector<arma::mat> Bm_g(N_g);
    
    for (int i = 0; i < N_g; i++) {
      
      int ni = n_obs_g[i];
      arma::vec x_i(ni);
      
      x_i = Tg.rows(0, ni-1).col(i);
      
      arma::mat B_i = bs_rcpp(x_i, knots, degree,intercept,boundary);
      
      Bm_g[i] = B_i;
    }
    
    Bm[g] = Bm_g;
  }
  
  return Bm;
}


std::vector<arma::mat> build_Bm_single_group(
    const arma::mat & Tg, 
    const Rcpp::IntegerVector & n_obs,           
    const arma::vec & knots,
    int degree,
    bool intercept,
    Nullable<NumericVector> boundary) 
{
  // Input : 

  int N_g = Tg.n_cols;
  
  std::vector<arma::mat> Bm(N_g);
    
  for (int i = 0; i < N_g; i++) {
      
    int ni = n_obs[i];
    arma::vec x_i(ni);
      
    x_i = Tg.rows(0, ni-1).col(i);
      
    arma::mat B_i = bs_rcpp(x_i, knots, degree,intercept,boundary);
      
    Bm[i] = B_i;
  }
  
  return Bm;
}

arma::mat build_Bm_individual(
    const arma::vec& time_i,      
    const arma::vec& knots,
    int degree,
    bool intercept,
    Rcpp::Nullable<Rcpp::NumericVector> boundary = R_NilValue)
{
  
  arma::mat B_i = bs_rcpp(
    time_i,
    knots,
    degree,
    intercept,
    boundary
  );
  
  return B_i;
}

arma::vec check_mean_param(Rcpp::Nullable<arma::vec> v,int p, std::string param) 
{

  if (v.isNull()) {
    return arma::vec(p, arma::fill::zeros);
  }
  
  arma::vec v_rcpp = Rcpp::as<arma::vec>(v);
  
  if (v_rcpp.n_elem != static_cast<arma::uword>(p))
  {
    Rcpp::stop("Vector param '%s' has wrong length: expected %d, got %d",
               param, p, v_rcpp.size());
  }
  
  return v_rcpp;
}

arma::vec compute_Upsilon(int order, int n_knots_h)
{
  
  // nu = c(rep(0,order), middle knots, rep(1,order))
  arma::vec nu(2 * order + n_knots_h, arma::fill::zeros);
  
  // middle knots: seq(0,1,len=n_knots_h+2)[-c(1,last)]
  arma::vec seq_full = arma::linspace(0.0, 1.0, n_knots_h + 2);
  nu.subvec(order, order + n_knots_h - 1) =
    seq_full.subvec(1, n_knots_h);
  
  // last rep(1, order)
  nu.subvec(order + n_knots_h, 2 * order + n_knots_h - 1).ones();
  
  // Upsilon length = n_knots_h + order
  arma::vec Upsilon(n_knots_h + order);
  
  Upsilon(0) = (nu(order - 1) - nu(0)) / (order - 1.0);
  
  for (int i = 0; i < n_knots_h + order - 1; i++) {
    Upsilon(i + 1) =
      (nu(i + order) - nu(i + 1)) / (order - 1.0) + Upsilon(i);
  }
  
  return Upsilon;
}

arma::vec check_mean_param_gaussian(Rcpp::Nullable<arma::vec> v,int q, int order, int n_knots_h,std::string param) 
{

  if (v.isNull()) {
    arma::vec phi_0 = compute_Upsilon(order, n_knots_h);
    return phi_0;
  }
  
  arma::vec v_rcpp = Rcpp::as<arma::vec>(v);
  
  if (v_rcpp.n_elem != static_cast<arma::uword>(q))
  {
    Rcpp::stop("Vector param '%s' has wrong length: expected %d, got %d",
               param, q, v_rcpp.size());
  }
  
  return v_rcpp;
}

arma::mat init_a_c_save(int nrun, int n_patients, double mu, double sigma) 
{
  
  arma::mat a_c_save(nrun, n_patients, arma::fill::value(arma::datum::nan));

  for(int i = 0; i < n_patients; i++) {
    a_c_save(0, i) = rnorm_1sample(mu, sigma);
  }
  
  return a_c_save;

}

std::vector<arma::mat> init_beta_save_group(int G,int nrun,int p, const arma::vec& beta_0,const arma::mat& Sigma) 
{
  std::vector<arma::mat> beta_save_group(G);
  
  for (int g = 0; g < G; g++) {
    
    arma::mat beta_mat(nrun, p, arma::fill::value(arma::datum::nan));
    
    beta_mat.row(0) = rmvnorm_1sample(beta_0, Sigma).t();
    
    beta_save_group[g] = beta_mat;
  }
  
  return beta_save_group;
}

arma::mat init_beta_save(int nrun,int p, const arma::vec& beta_0,const arma::mat& Sigma) 
{
    arma::mat beta_mat(nrun, p, arma::fill::value(arma::datum::nan));
    
    beta_mat.row(0) = rmvnorm_1sample(beta_0, Sigma).t();
    
    return beta_mat;
}

arma::mat build_beta_current(const std::vector<arma::mat> & beta_save_group, int p)
{
  
  int G = beta_save_group.size();
  
  arma::mat beta_current(p, G, arma::fill::none);
  
  for (int g = 0; g < G; g++) {
    arma::mat beta_save_g = beta_save_group[g];
    beta_current.col(g) = beta_save_g.row(0).t();
  }
  
  return beta_current;
}

std::vector<std::vector<arma::mat>> init_gamma_save_group(int G,const arma::ivec& n_per_group,int nrun,int k,
                                      const arma::vec& gamma_0,const arma::mat& sigma_gamma_matrix)
{
  std::vector<std::vector<arma::mat>> gamma_save_group(G);
  
  for (int g = 0; g < G; g++) {
    std::vector<arma::mat> gamma_save_gruppo_G(n_per_group[g]);
    
    for (int i = 0; i < n_per_group[g]; i++) {
      arma::mat gamma_save_ind(nrun, k, arma::fill::value(arma::datum::nan));
      
      gamma_save_ind.row(0) = rmvnorm_1sample(gamma_0, sigma_gamma_matrix).t();
      
      gamma_save_gruppo_G[i] = gamma_save_ind;
    }
    
    gamma_save_group[g] = gamma_save_gruppo_G;
  }
  
  return gamma_save_group;
}

std::vector<arma::mat> build_gamma_current(const std::vector<std::vector<arma::mat>> & gamma_save_group,
                                           const Rcpp::IntegerVector& n_per_group,int k)
{
  
  int G = gamma_save_group.size();
  std::vector<arma::mat> gamma_current(G);
  
  for (int g = 0; g < G; g++) {
    
    std::vector<arma::mat> gamma_save_g = gamma_save_group[g];
    int n_g = n_per_group[g];
    
    arma::mat gamma_g(k, n_g, arma::fill::none);
    
    for (int i = 0; i < n_g; i++) {
      
      arma::mat gamma_save_ind = gamma_save_g[i];
      
      gamma_g.col(i) = gamma_save_ind.row(0).t();
    }
    
    gamma_current[g] = gamma_g;
  }
  
  return gamma_current;
}

arma::mat init_csi_ind(int nrun, int q, bool INIT_WARP,
                       Rcpp::Nullable<arma::vec> phi_init,
                       double SOMMA_csi, const arma::vec& a_f, double b_f) 
{
    // matrice di output con NaN
    arma::mat csi_save_ind(nrun, q, arma::fill::value(arma::datum::nan));

    if (INIT_WARP) {
        if (phi_init.isNull()) {
            Rcpp::stop("phi_init must be provided when INIT_WARP = TRUE");
        }

        arma::vec phi = Rcpp::as<arma::vec>(phi_init); 

        check_phi_warping(phi); 

        csi_save_ind(0, 0) = SOMMA_csi * phi[0];
        for (int j = 1; j < q; j++) {
            csi_save_ind(0, j) = SOMMA_csi * (phi[j] - phi[j - 1]);
        }
    } else {
        csi_save_ind(0, 0) = 0.0;
        for (int j = 1; j < q; j++) {
            csi_save_ind(0, j) = rgamma_1sample(a_f[j - 1], 1.0 / b_f); // scale = 1/rate
        }
    }

    return csi_save_ind;
}

std::vector<std::vector<arma::mat>> init_csi_save_group(int G,const arma::ivec& n_per_group,int nrun,int q,
                               bool INIT_WARP, Nullable<arma::vec> phi_init,double SOMMA_csi,
                               const arma::vec& a_f,double b_f)
{
  std::vector<std::vector<arma::mat>> csi_save_group(G);
  
  for (int g = 0; g < G; g++) {
    int N = n_per_group[g];
    std::vector<arma::mat> group_list(N);
    
    for (int i = 0; i < N; i++) {
      group_list[i] = init_csi_ind(nrun, q, INIT_WARP, phi_init, SOMMA_csi, a_f, b_f);
    }
    
    csi_save_group[g] = group_list;
  }
  
  return csi_save_group;
}

std::vector<arma::mat> init_csi_save(int n_pat,int nrun,int q,
                               bool INIT_WARP, Nullable<arma::vec> phi_init,double SOMMA_csi,
                               const arma::vec& a_f,double b_f)
{
  std::vector<arma::mat> csi_save_list(n_pat);
    
  for (int i = 0; i < n_pat; i++) {
      csi_save_list[i] = init_csi_ind(nrun, q, INIT_WARP, phi_init, SOMMA_csi, a_f, b_f);
  }
  
  return csi_save_list;
}

arma::vec check_af(Nullable<arma::vec> a_f,int q)
{
  
  if (a_f.isNull()) {
    Rcpp::stop("a_f must not be NULL");
  }
  
  arma::vec a_f_vec = Rcpp::as<arma::vec>(a_f);

  if(a_f_vec.n_elem != static_cast<arma::uword>(q-1)){
    Rcpp::stop("a_f does not match the correct length: " + std::to_string(q-1));
  }

  return a_f_vec;
}

arma::mat init_phi_ind_gamma_adapt(int nrun, const arma::rowvec& csi_save_ind)
{
  int q = csi_save_ind.n_elem;
  arma::mat phi_save_ind(nrun, q, arma::fill::value(arma::datum::nan));
  
  double sum_csi = arma::sum(csi_save_ind);
  phi_save_ind.row(0) = arma::cumsum(csi_save_ind) / sum_csi;
  
  return phi_save_ind;
}

std::vector<std::vector<arma::mat>> init_phi_save_group_gamma_adapt(int G,const arma::ivec& n_per_group,int nrun,
                               const std::vector<std::vector<arma::mat>>& csi_save_group)
{
  std::vector<std::vector<arma::mat>> phi_save_group(G);
  
  for (int g = 0; g < G; g++) {
    int N = n_per_group[g];
    std::vector<arma::mat> group_list(N);
    std::vector<arma::mat> csi_save_g = csi_save_group[g];
    
    for (int i = 0; i < N; i++) {
      arma::mat csi_mat = csi_save_g[i];
      arma::rowvec csi_row = csi_mat.row(0);
      
      group_list[i] = init_phi_ind_gamma_adapt(nrun, csi_row);  
    }
    
    phi_save_group[g] = group_list;
  }
  
  return phi_save_group;
}

std::vector<arma::mat> init_phi_save_gamma_adapt(int n_pat,int nrun,
                               const std::vector<arma::mat>& csi_save)
{
  std::vector<arma::mat> phi_save(n_pat);
    
  for (int i = 0; i < n_pat; i++) {
    arma::mat csi_mat = csi_save[i];
    arma::rowvec csi_row = csi_mat.row(0);
      
    phi_save[i] = init_phi_ind_gamma_adapt(nrun, csi_row);  
  }
  
  return phi_save;
}

std::vector<std::vector<arma::mat>> init_phi_save_group_gaussian(int G,const arma::ivec& n_per_group,int nrun,int q,
                                                                 const arma::vec& phi_0,const arma::mat& Sigma_phi,
                                                    bool INIT_WARP, Nullable<arma::vec> phi_init) 
{
  std::vector<std::vector<arma::mat>> phi_save_group(G);
  
  for (int g = 0; g < G; g++) {
    
    int N = n_per_group[g];
    std::vector<arma::mat> group_list(N);
    
    for (int i = 0; i < N; i++) {
      
      arma::mat phi_mat(nrun, q, arma::fill::value(arma::datum::nan));
      
      if(INIT_WARP) {
        if (phi_init.isNull()) {
          Rcpp::stop("phi_init must be provided when INIT_WARP = TRUE");
        }
        
        check_phi_warping(Rcpp::as<arma::vec>(phi_init)); 

        phi_mat.row(0) = Rcpp::as<arma::vec>(phi_init).t();
        
      } else {   
        phi_mat.row(0) = rmvnorm_1sample(phi_0, Sigma_phi).t();
      }
      
      group_list[i] = phi_mat;  
    }

    phi_save_group[g] = group_list;
  }
  
  return phi_save_group;
}

std::vector<arma::mat> init_phi_save_gaussian(int N,int nrun,int q,
                                              const arma::vec& phi_0,const arma::mat& Sigma_phi,
                                              bool INIT_WARP, Nullable<arma::vec> phi_init) 
{
    
    std::vector<arma::mat> phi_list(N);
    
    for (int i = 0; i < N; i++) {
      
      arma::mat phi_mat(nrun, q, arma::fill::value(arma::datum::nan));
      
      if(INIT_WARP) {
        if (phi_init.isNull()) {
          Rcpp::stop("phi_init must be provided when INIT_WARP = TRUE");
        }
        
        phi_mat.row(0) = Rcpp::as<arma::vec>(phi_init).t();
        
      } else {   
        phi_mat.row(0) = rmvnorm_1sample(phi_0, Sigma_phi).t();
      }
      
      phi_list[i] = phi_mat;  
  }
  
    return phi_list;
}

arma::mat init_x_ind(int nrun, const arma::rowvec& csi_save_ind)
{
  // This function returns the initialization of x for one individual
  // Input : nrun = number of runs 
  //         csi_save_ind = matrix which rows are the csi vector for that iteration for one individual 
  // Output: the x matrix (nrun x q) for one individual 
  int q = csi_save_ind.n_elem;
  arma::mat x_save_ind(nrun, q, arma::fill::value(arma::datum::nan));
  
  x_save_ind(0, 0) = 0.0;
  for(int j = 1; j < q; j++) {
    x_save_ind(0, j) = std::log(csi_save_ind[j]);
  }
  
  return x_save_ind;
}

std::vector<std::vector<arma::mat>> init_x_save_group(int G, const arma::ivec& n_per_group, int nrun,
                             const std::vector<std::vector<arma::mat>>& csi_save_group) 
{ 
  // This function initializes the x_save_group object
  // Input : G =number of groups 
  //         n_per_group = vector with the number of inididulas for each group as elements
  //         nrun = number of runs 
  //         csi_save_group = list of lists containing the csi elements 
  // Output: List of lists whose elements are matrices whose rows are the x elements at the t-th iteration
  

  std::vector<std::vector<arma::mat>> x_save_group(G);
  
  for(int g = 0; g < G; g++) {
    int N = n_per_group[g];
    std::vector<arma::mat> group_list(N);
    std::vector<arma::mat> csi_save_group_g = csi_save_group[g];
    
    for(int i = 0; i < N; i++) {
      
      arma::mat csi_mat = csi_save_group_g[i];
      arma::rowvec csi_row = csi_mat.row(0);
      
      group_list[i] = init_x_ind(nrun, csi_row);
    }
    
    x_save_group[g] = group_list;
  }
  
  return x_save_group;
}

std::vector<arma::mat> init_x_save(int n_pat, int nrun,
                             const std::vector<arma::mat>& csi_save) 
{ 
  // This function initializes the x_save_group object
  // Input : G =number of groups 
  //         n_per_group = vector with the number of inididulas for each group as elements
  //         nrun = number of runs 
  //         csi_save_group = list of lists containing the csi elements 
  // Output: List of lists whose elements are matrices whose rows are the x elements at the t-th iteration

  std::vector<arma::mat> x_save(n_pat);
    
  for(int i = 0; i < n_pat; i++) {
      arma::mat csi_mat = csi_save[i];
      arma::rowvec csi_row = csi_mat.row(0);
      
      x_save[i] = init_x_ind(nrun, csi_row);
  }
  
  return x_save;
}

arma::mat check_zeta0(int q, double zeta_0_coeff,Rcpp::Nullable<arma::mat> zeta0) 
{ 
  // This function checks if the user has provided a zeta matrix and creates it otherwise
  // Input : q = dimension of the warping vector and so the dimension of the zeta matrix is q-1
  //         zeta_0_coeff = value s. t. zeta = I(q-1,q-1) * zeta_0_coeff
  //         zeta_0 = zeta matrix. If null zeta = I(q-1,q-1) * zeta_0_coeff
  // Output: zeta matrix  
  if (zeta0.isNotNull()) {
    Rcpp::NumericMatrix Z(zeta0);
    arma::mat Z_arma(Z.begin(), Z.nrow(), Z.ncol(), false);
   
    if (Z_arma.n_rows != Z_arma.n_cols) {
      Rcpp::stop("Matrix zeta_0 must be square");
    }
    
    if (Z_arma.n_rows != static_cast<arma::uword>(q-1)) {
      Rcpp::stop("Matrix zeta_0 must be of dimension " + std::to_string(q-1) + " x " + std::to_string(q-1));
    }
    
    if (!Z_arma.is_sympd()) {
      Rcpp::stop("Matrix zeta_0 is not symmetric positive definite");
    }
    
    return Z_arma; 
  } else {
    return arma::eye(q-1, q-1) * zeta_0_coeff;
  }
}

std::vector<std::vector<arma::mat>> init_zeta_group(int G,const arma::ivec& n_per_group,int q, double zeta_0_coeff,
                                                    Rcpp::Nullable<arma::mat> zeta_0 = R_NilValue) 
{ 
  // This function initializes the zeta_group object
  // Input : G =number of groups 
  //         n_per_group = vector with the number of inididulas for each group as elements
  //         q = dimension of the warping vector and so the dimension of the zeta matrix is q-1
  //         zeta_0_coeff = value s. t. zeta = I(q-1,q-1) * zeta_0_coeff
  //         zeta_0 = zeta matrix. If null zeta = I(q-1,q-1) * zeta_0_coeff
  // Output: List of lists whose elements are the zeta matrices 
  
  arma::mat zeta_mat = check_zeta0(q, zeta_0_coeff, zeta_0);
  
  std::vector<std::vector<arma::mat>> zeta_group(G);
  
  for (int g = 0; g < G; g++) {
    int N = n_per_group[g];
    std::vector<arma::mat> group_list(N);
    
    for (int i = 0; i < N; i++) {
      group_list[i] = zeta_mat;
    }
    
    zeta_group[g] = group_list;
  }
  
  return zeta_group;
}

std::vector<arma::mat> init_zeta(int n_pat,int q, double zeta_0_coeff,
                                                    Rcpp::Nullable<arma::mat> zeta_0 = R_NilValue) 
{ // This function initializes the zeta_group object
  // Input : G =number of groups 
  //         n_per_group = vector with the number of inididulas for each group as elements
  //         q = dimension of the warping vector and so the dimension of the zeta matrix is q-1
  //         zeta_0_coeff = value s. t. zeta = I(q-1,q-1) * zeta_0_coeff
  //         zeta_0 = zeta matrix. If null zeta = I(q-1,q-1) * zeta_0_coeff
  // Output: List of lists whose elements are the zeta matrices 
  
  arma::mat zeta_mat = check_zeta0(q, zeta_0_coeff, zeta_0);
  
  std::vector<arma::mat> zeta_save(n_pat);
    
  for (int i = 0; i < n_pat; i++) {
      zeta_save[i] = zeta_mat;
  }
    
  return zeta_save;
}

double check_sd0(int q, Rcpp::Nullable<double> sd0_user, double sd_min, double sd_max)
{
  if (sd0_user.isNotNull()) {
    double sd0_final = rho_clipping(as<double>(sd0_user),sd_min,sd_max);
    
    if(sd0_final != as<double>(sd0_user)){
      Rcpp::Rcout << "Warning: sd0 (" << Rcpp::as<double>(sd0_user)
                  << ") has been adjusted to " << sd0_final
                  << " to fit within the allowed range [" << sd_min 
                  << ", " << sd_max << "]." << std::endl;
    }
    
    return sd0_final;
    
  } else {
    return rho_clipping( (2.34 * 2.34) / (q - 1),sd_min,sd_max);
  }
}

std::vector<std::vector<arma::vec>> init_sd_save_group(int G,const arma::ivec& n_per_group,int nrun,int q,
                                                       Nullable<double> sd0,double sd_min,double sd_max) 
{
  // crea una lista di loste (grppi individui) con vettori lunghi nrun con dentro sd0. il pirmo valore viene messo 
  
  double sd_final = check_sd0(q,sd0,sd_min,sd_max);
  
  std::vector<std::vector<arma::vec>> sd_group(G);
  
  for (int g = 0; g < G; g++) {
    int N = n_per_group[g];
    std::vector<arma::vec> group_list(N);
    
    for (int i = 0; i < N; i++) {
      arma::vec vec(nrun, arma::fill::value(arma::datum::nan));
      vec[0] = sd_final;                           
      group_list[i] = vec;
    }
    
    sd_group[g] = group_list;
  }
  
  return sd_group;
}

std::vector<arma::vec> init_sd_save(int n_pat,int nrun,int q,
                                    Nullable<double> sd0,double sd_min,double sd_max) 
{
  // crea una lista di loste (grppi individui) con vettori lunghi nrun con dentro sd0. il pirmo valore viene messo 
  
  double sd_final = check_sd0(q,sd0,sd_min,sd_max);
  
  std::vector<arma::vec> sd_save(n_pat);
    
  arma::vec vec(nrun, arma::fill::value(arma::datum::nan));
  vec[0] = sd_final; 

  for (int i = 0; i < n_pat; i++) {                          
      sd_save[i] = vec;
  }
    

  return sd_save;
}

arma::vec allocate_obs(const arma::vec& t, int n_obs)
{
    int L = t.n_elem - 1;
    arma::vec obs_counts(L);

    // calcola intervalli
    arma::vec intervals = t.tail(L) - t.head(L); // t[1:] - t[0:L-1]

    // proporzioni
    arma::vec prop = intervals / arma::sum(intervals);

    // assegnazione
    obs_counts = n_obs * prop;

    // se vuoi interi, arrotonda
    obs_counts = arma::round(obs_counts);

    // correzione per garantire somma esatta
    int diff = n_obs - arma::accu(obs_counts);
    if (diff != 0) {
        // aggiungi/subtrai 1 dal primo elemento per correggere
        obs_counts[0] += diff;
    }

    return obs_counts;
}

arma::mat update_zeta(const arma::mat& x_mat,double sd_iter,int iter,int q,double EPS) 
{
  
  arma::mat X = x_mat.submat(0, 1, iter-1, q-1);
  
  arma::mat C = Rcpp::as<arma::mat>(cov_fun(X));
  
  arma::mat D(q - 1, q - 1, arma::fill::zeros);
  D.diag().fill(EPS);
  
  arma::mat out = sd_iter * (C + D);
  
  return out;
}

arma::vec build_csi_from_x(const arma::vec& x, int q)
{
  
  Rcpp::NumericVector out(q);
  out[0] = 0.0;
  
  for (int j = 1; j < q; ++j) {
    out[j] = std::exp(x[j]);
  }
  
  return out;
}

arma::vec build_phi_from_csi(const arma::vec& csi, int q)
{
  
  double tot = arma::accu(csi);
 
  arma::vec phi = arma::cumsum(csi) / tot;
  
  return phi;
}

Rcpp::List normalize_accepts(const std::vector<Rcpp::IntegerVector> & accepts, int nrun)
{
  
  int n_groups = accepts.size();
  Rcpp::List accepts_norm(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    NumericVector acc_g = Rcpp::as<NumericVector>(accepts[g]);
    accepts_norm[g] = acc_g / (nrun-1) ;
  }
  
  return accepts_norm;
}

Rcpp::List group_term_post_mean(const std::vector<arma::mat> & beta_save_group, int nburn)
{
  
  int n_groups = beta_save_group.size();
  Rcpp::List beta_post_group(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    arma::mat beta_save_g = beta_save_group[g];
    
    int nrun = beta_save_g.n_rows;
    
    arma::mat beta_post = beta_save_g.rows(nburn, nrun - 1);

    arma::vec beta_mean = arma::mean(beta_post, 0).t();
    
    beta_post_group[g] = beta_mean;
  }
  
  return beta_post_group;
}

Rcpp::List individual_term_post_mean(
    const std::vector<std::vector<arma::mat>> & gamma_save_group,
    const Rcpp::IntegerVector& n_per_group,
    int nburn
)
{
  
  int n_groups = gamma_save_group.size();
  Rcpp::List gamma_post_group(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    std::vector<arma::mat> gamma_save_g = gamma_save_group[g];
    int n_g = n_per_group[g];
    
    Rcpp::List gamma_post_g(n_g);
    
    for (int i = 0; i < n_g; i++) {
      
      arma::mat gamma_save_gi = gamma_save_g[i];
      
      int nrun = gamma_save_gi.n_rows;
      if (nburn >= nrun) {
        Rcpp::stop("nburn must be smaller than number of iterations");
      }
      
      arma::mat gamma_post = gamma_save_gi.rows(nburn, nrun - 1);
      
      // colMeans
      arma::vec gamma_mean = arma::mean(gamma_post, 0).t();
      
      gamma_post_g[i] = gamma_mean;
    }
    
    gamma_post_group[g] = gamma_post_g;
  }
  
  return gamma_post_group;
}

Rcpp::List individual_term_post_mean_sd(
    const std::vector<std::vector<arma::vec>> & gamma_save_group,
    const Rcpp::IntegerVector& n_per_group,
    int nburn)
{
  
  int n_groups = gamma_save_group.size();
  Rcpp::List gamma_post_group(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    const std::vector<arma::vec>& gamma_save_g = gamma_save_group[g];
    int n_g = n_per_group[g];
    
    Rcpp::List gamma_post_g(n_g);
    
    for (int i = 0; i < n_g; i++) {
      
      const arma::vec& gamma_save_gi = gamma_save_g[i];
      
      int nrun = gamma_save_gi.n_elem;
      if (nburn >= nrun) {
        Rcpp::stop("nburn must be smaller than number of iterations");
      }
      
      arma::vec gamma_post = gamma_save_gi.subvec(nburn, nrun - 1);
      
      // ✅ media come double
      double gamma_mean = arma::mean(gamma_post);
      
      gamma_post_g[i] = gamma_mean;
    }
    
    gamma_post_group[g] = gamma_post_g;
  }
  
  return gamma_post_group;
}

Rcpp::List individual_term_post_mean_sd_single_group(
    const std::vector<arma::vec> & sd_save,
    int n_pat, int nburn) 
{
    
    Rcpp::List sd_post(n_pat);
    
    for (int i = 0; i < n_pat; i++) {
      
      arma::vec sd_save_i = sd_save[i];
      
      int nrun = sd_save_i.n_elem;  // <-- usare n_elem per arma::vec
      if (nburn >= nrun) {
        Rcpp::stop("nburn must be smaller than number of iterations");
      }
      
      arma::vec sd_post_vec = sd_save_i.subvec(nburn, nrun - 1);  // <-- subvec
      
      double sd_mean = arma::mean(sd_post_vec);  // più chiaro di accu/n_elem
      
      sd_post[i] = sd_mean;  // assegna correttamente alla lista Rcpp
    }
  
    return sd_post;
}

Rcpp::List build_h_post(
    const std::vector<std::vector<arma::mat>> & Bh_group,          
    const List & phi_post_group,
    const Rcpp::IntegerVector& n_per_group)
{
  
  int n_groups = Bh_group.size();
  Rcpp::List h_p(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    int n_i = n_per_group[g];
    Rcpp::List h_p_g(n_i);
    
    std::vector<arma::mat> Bh_g  = Bh_group[g];
    std::vector<arma::mat> phi_g = phi_post_group[g];
    
    for (int i = 0; i < n_i; i++) {
      
      arma::mat Bh_gi  = Bh_g[i];
      arma::vec phi_gi = phi_g[i];
      
      arma::vec h_pred_gi = Bh_gi * phi_gi;
      
      h_p_g[i] = h_pred_gi;
    }
    
    h_p[g] = h_p_g;
  }
  
  return h_p;
}

Rcpp::List build_h_post_single_group(
    const std::vector<arma::mat> & Bh,          
    const List & phi_post,
    double n_pat)
{
  
  Rcpp::List h_p(n_pat);
    
  for (int i = 0; i < n_pat; i++) {
      
      arma::mat Bh_gi  = Bh[i];
      arma::vec phi_gi = phi_post[i];
      
      arma::vec h_pred_gi = Bh_gi * phi_gi;
      
      h_p[i] = h_pred_gi;
  }
  
  return h_p;
}

std::vector<arma::mat> build_h_post_cpp(
    const std::vector<std::vector<arma::mat>> & Bh_group,          
    const List & phi_post_group,
    const Rcpp::IntegerVector& n_per_group,
    int n_obs_max)
{
  
  int n_groups = Bh_group.size();
  std::vector<arma::mat> h_p(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    int n_i = n_per_group[g];
    std::vector<arma::mat> Bh_g  = Bh_group[g];
    List phi_g = phi_post_group[g];
    
    // ---- matrice h_g: (n_obs_max x n_i), inizializzata a NA
    arma::mat h_g(n_obs_max, n_i);
    h_g.fill(arma::datum::nan);
    
    // ---- riempi colonne
    for (int i = 0; i < n_i; i++) {
      
      arma::mat Bh_gi  = Bh_g[i];
      arma::vec phi_gi = phi_g[i];
      
      arma::vec h_gi = Bh_gi * phi_gi;  // lunghezza n_obs_gi
      
      h_g.submat(0, i, h_gi.n_elem - 1, i) = h_gi;
    }
    
    h_p[g] = h_g;
  }
  
  return h_p;
}

arma::mat build_h_post_cpp_single_group(
    const std::vector<arma::mat> & Bh_group,          
    const List & phi_post_group,
    int n_pat, int n_obs_max)
{
    
    arma::mat h_g(n_obs_max, n_pat);
    h_g.fill(arma::datum::nan);
    
    for (int i = 0; i < n_pat; i++) {
      
      arma::mat Bh_gi  = Bh_group[i];
      arma::vec phi_gi = phi_post_group[i];
      
      arma::vec h_gi = Bh_gi * phi_gi;  // lunghezza n_obs_gi
      
      h_g.submat(0, i, h_gi.n_elem - 1, i) = h_gi;
    }
    
  return h_g;
}

Rcpp::List build_y_pred_gardella(
    const std::vector<std::vector<arma::mat>> & Bm_post_beta_group,   // list[g][i] -> arma::mat
    const List & beta_post_group,      // list[g] -> arma::vec
    const std::vector<std::vector<arma::mat>> & Bm_post_gamma_group,  // list[g][i] -> arma::mat
    const List & gamma_post_group,     // list[g][i] -> arma::vec
    const Rcpp::IntegerVector& n_per_group) 
{
  
  int n_groups = Bm_post_beta_group.size();
  Rcpp::List y_p(n_groups);
  
  for (int g = 0; g < n_groups; g++) {
    
    int n_i = n_per_group[g];
    Rcpp::List y_p_g(n_i);
    
    arma::vec beta_g = beta_post_group[g];
    std::vector<arma::mat> Bm_beta_g  = Bm_post_beta_group[g];
    std::vector<arma::mat> Bm_gamma_g = Bm_post_gamma_group[g];
    List gamma_g    = gamma_post_group[g];
    
    for (int i = 0; i < n_i; i++) {
      
      arma::mat Bm_beta_gi  = Bm_beta_g[i];
      arma::mat Bm_gamma_gi = Bm_gamma_g[i];
      arma::vec gamma_gi    = Rcpp::as<arma::vec>(gamma_g[i]);
      
      arma::vec y_pred_gi =  Bm_beta_gi  * beta_g +  Bm_gamma_gi * gamma_gi;
      
      y_p_g[i] = y_pred_gi;
    }
    
    y_p[g] = y_p_g;
  }
  
  return y_p;
}

Rcpp::List build_y_pred_telesca(
    const std::vector<arma::mat> & Bm_post_beta,   // list[g][i] -> arma::mat
    const arma::vec & beta_post,      // list[g] -> arma::vec
    const Rcpp::List& a_post,
    const Rcpp::List& c_post,
    int n_pat) 
{

  Rcpp::List y_p(n_pat);

  for (int i = 0; i < n_pat; i++) {
      
      arma::mat Bm_beta_gi  = Bm_post_beta[i];
      double a_i = a_post[i];
      double c_i = c_post[i];
      
      arma::vec y_pred_gi = a_i *  Bm_beta_gi  * beta_post +  c_i;
      
      y_p[i] = y_pred_gi;
    }
  
  return y_p;
}

List gardella_gamma_adaptation(
    std::vector<arma::mat> y,Nullable<int> n_groups,Nullable<IntegerVector> n_per_group,
    int nburn,int niter,int n_knots_beta,int n_knots_gamma,int n_knots_phi,
    Nullable<arma::vec> beta0, Nullable<arma::vec> gamma0,
    double a_eps,double b_eps,double a_lambda,double b_lambda,double a_gamma,
    double b_gamma, Nullable<arma::vec> a_f, double b_f,
    int degree_beta, int degree_gamma, int degree_phi,
    bool intercept_beta,bool intercept_gamma, bool intercept_phi,
    Nullable<NumericVector> boundary_beta, Nullable<NumericVector> boundary_gamma,Nullable<NumericVector> boundary_phi,
    Nullable<arma::mat> zeta_0, double zeta_0_coeff,
    Nullable<double> sd0,double sd_min,double sd_max,double target_alpha,double LAMB,
    double EPS,int n0_zeta,bool INIT_WARP,Nullable<arma::vec> phi_init,double SOMMA_csi)
{
  
  // Check sd_min < sd_max 
  check_rho_clipping(sd_min,sd_max);
  
  // Extract and check the information from y
  List y_info = extract_y_info(y);
  check_params_y(y_info,n_groups,n_per_group);
  
  Rcpp::IntegerVector n_pg = Rcpp::as<Rcpp::IntegerVector>(y_info["n_per_group"]);
  int G = Rcpp::as<int>(y_info["n_groups"]);
  std::vector<Rcpp::IntegerVector> n_obs = y_info["n_obs"];
  std::vector<arma::vec> y_val = y_info["y_val"]; 
  int n_patients = Rcpp::as<int>(y_info["n_patients"]);
  arma::vec n_obs_tot_per_group = y_info["n_obs_tot_per_group"];
  int tot_obs = arma::accu(n_obs_tot_per_group);
  
  // Define the spline parameters
  List beta_spline  = compute_spline_order(degree_beta, intercept_beta, std::string("beta"));
  List gamma_spline = compute_spline_order(degree_gamma, intercept_gamma, std::string("gamma"));
  List phi_spline   = compute_spline_order(degree_phi, intercept_phi, std::string("phi"));
  
  intercept_beta  = beta_spline["intercept"];
  intercept_gamma = gamma_spline["intercept"];
  intercept_phi   = phi_spline["intercept"];
  
  int p = n_knots_beta + Rcpp::as<int>(beta_spline["order_spline"]);
  int k = n_knots_gamma + Rcpp::as<int>(gamma_spline["order_spline"]);
  int q = n_knots_phi + Rcpp::as<int>(phi_spline["order_spline"]);

  // Check a_f 
  arma::vec a_f_vec = check_af(a_f,q);
  
  arma::vec knots_beta  = build_knots(n_knots_beta);
  arma::vec knots_gamma = build_knots(n_knots_gamma);
  arma::vec knots_phi   = build_knots(n_knots_phi);
  
  std::vector<arma::mat> h_list = build_time_matrix(n_obs,y_info["n_obs_max"]);
  
  std::vector<std::vector<arma::mat>> Bm_beta  = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_beta,degree_beta,intercept_beta,boundary_beta);
  std::vector<std::vector<arma::mat>> Bm_gamma = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_gamma,degree_gamma,intercept_gamma,boundary_gamma);
  std::vector<std::vector<arma::mat>> Bm_phi   = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_phi,degree_phi,intercept_phi,boundary_phi);
  
  // Stacked splines
  std::vector<arma::mat> Bm_beta_stacked = stack_Matrix_groups(Bm_beta,p);
  
  // Number of iterations 
  int nrun = nburn + niter;
  
  // Define the parameters 
  
  // Sigma epsilon
  arma::vec sigma_eps_save(nrun);
  sigma_eps_save[0] = rinvgamma_1sample(a_eps,b_eps);
  
  // Lambda 
  arma::vec lambda_save(nrun);
  lambda_save[0] = rinvgamma_1sample(a_lambda,b_lambda);
  
  // Sigma gamma 
  arma::vec sigma_gamma_save(nrun);
  sigma_gamma_save[0] = rinvgamma_1sample(a_gamma,b_gamma);
  
  // Beta 
  arma::vec beta_0 = check_mean_param(beta0,p,std::string("beta"));
  arma::mat Omega = omega_P_cpp(p);  
  arma::mat Sigma_beta = arma::inv_sympd(Omega / lambda_save[0]);
  
  std::vector<arma::mat> beta_save_group = init_beta_save_group(G,nrun,p,beta_0,Sigma_beta);
  arma::mat beta_current = build_beta_current(beta_save_group,p);
  
  // Gamma 
  arma::vec gamma_0 = check_mean_param(gamma0,k,std::string("gamma"));
  
  arma::mat sigma_gamma_matrix = arma::diagmat(arma::vec(k, arma::fill::ones) * sigma_gamma_save[0]);
  arma::mat inv_sigma_gamma_matrix = arma::inv(sigma_gamma_matrix);
  
  std::vector<std::vector<arma::mat>> gamma_save_group = init_gamma_save_group(G,n_pg,nrun,k,gamma_0,sigma_gamma_matrix);
  std::vector<arma::mat> gamma_current = build_gamma_current(gamma_save_group,n_pg,k);
  
  // Csi 
  std::vector<std::vector<arma::mat>> csi_save_group = init_csi_save_group(G,n_pg,nrun,q,INIT_WARP,phi_init,SOMMA_csi,a_f_vec,b_f);

  // Phi 
  std::vector<std::vector<arma::mat>> phi_save_group = init_phi_save_group_gamma_adapt(G,n_pg,nrun,csi_save_group);
  
  // x = log(csi)
  std::vector<std::vector<arma::mat>> x_save_group = init_x_save_group(G,n_pg,nrun,csi_save_group);
  
  // MH parameters 
  // Zeta 
  std::vector<std::vector<arma::mat>> zeta_group = init_zeta_group(G,n_pg,q,zeta_0_coeff,zeta_0);
  
  // sd 
  std::vector<std::vector<arma::vec>> sd_save_group = init_sd_save_group(G,n_pg,nrun,q,sd0,sd_min,sd_max);
  
  // Acceptance rates 
  std::vector<Rcpp::IntegerVector> accepts(G);
  
  for (int g = 0; g < G; g++) {
    accepts[g] = Rcpp::IntegerVector(n_pg[g], 0);
  }

  arma::vec x_new(q);
  int accettato;
  double alpha;
  
 
  RObject bar = txtProgressBar(_["min"] = 0, _["max"] = nrun-1, _["style"] = 3);
  
  for (int iter = 1; iter < nrun; iter++) {

    // w update 
    double w = std::pow(iter, -LAMB);

    double sum_lambda_term = 0.0; 
    double sum_sigma_gamma_term = 0.0;
    double sum_sigma_eps_term = 0.0;
    
    for(int g = 0; g < G; g++){

      arma::uword offset = 0;
      arma::vec c_gamma(y_val[g].n_elem, arma::fill::zeros); 
      
      for(int i = 0; i < n_pg[g]; i++ ){

        const arma::vec y_gi = y[g].col(i).subvec(0, n_obs[g][i] - 1);
        
        // MH step
        MH_gardella_gamma_adapt(x_new,alpha,accettato,x_save_group[g][i].row(iter-1).t(),zeta_group[g][i],
                                beta_save_group[g].row(iter-1).t(),
                                gamma_save_group[g][i].row(iter-1).t(),
                                sigma_eps_save[iter-1], a_f_vec, b_f,Bm_phi[g][i],y_gi,
                                degree_beta,degree_gamma,intercept_beta,intercept_gamma,
                                boundary_beta,boundary_gamma); 
        
        // Update acceptance rates 
        accepts[g][i] += accettato;
      
        x_save_group[g][i].row(iter) = x_new.t();
        
        // sd update;
        sd_save_group[g][i][iter] = rho_clipping(sd_save_group[g][i][iter-1] + 
                                      w * (alpha - target_alpha),sd_min,sd_max);
        
        // zeta update 
        if (iter > n0_zeta)
          {  
             zeta_group[g][i] = update_zeta(x_save_group[g][i],sd_save_group[g][i][iter],iter,q,EPS);
          }
        
        
        // Csi update
        csi_save_group[g][i].row(iter) = build_csi_from_x(x_save_group[g][i].row(iter).t(),q).t(); 
        
        // Phi update
        phi_save_group[g][i].row(iter) = build_phi_from_csi(csi_save_group[g][i].row(iter).t(),q).t();
        
        // h update
        arma::vec h_new = Bm_phi[g][i] * phi_save_group[g][i].row(iter).t();
        h_list[g].submat(0, i, n_obs[g][i]-1, i) = h_new;
        
        // Bm update 
        Bm_beta[g][i] = build_Bm_individual(h_new,knots_beta,degree_beta,intercept_beta,boundary_beta);
        Bm_gamma[g][i] = build_Bm_individual(h_new,knots_gamma,degree_gamma,intercept_gamma,boundary_gamma); 
        
        // Gamma update
        arma::mat V_gamma_gi = compute_post_cov_gamma(Bm_beta[g][i],beta_current.col(g),Bm_gamma[g][i],
                                                      y[g].col(i).subvec(0, n_obs[g][i] - 1),
                                                      sigma_gamma_save[iter-1],sigma_eps_save[iter-1]);
        arma::vec m_gamma_gi = compute_post_mean_gamma(Bm_beta[g][i],beta_current.col(g),Bm_gamma[g][i],y_gi,
                                                       sigma_gamma_save[iter-1],sigma_eps_save[iter-1],
                                                       gamma_0, V_gamma_gi);
        
        gamma_save_group[g][i].row(iter) = rmvnorm_1sample(m_gamma_gi,V_gamma_gi).t();
        
        gamma_current[g].col(i) = gamma_save_group[g][i].row(iter).t();

        arma::vec contrib_i = Bm_gamma[g][i] * gamma_current[g].col(i);
      
        c_gamma.subvec(offset, offset + contrib_i.n_elem - 1) = contrib_i;
        offset += contrib_i.n_elem;

        sum_sigma_eps_term += arma::accu(arma::square( y_gi - Bm_beta[g][i] * beta_current.col(g) - contrib_i ));

        
      }
      
      // Stacked matrix update
      Bm_beta_stacked[g] = stack_single_group(Bm_beta[g],p,n_obs[g]);
      
      // Beta update 
      arma::mat V_beta = compute_post_cov_beta(Bm_beta_stacked[g],Omega,lambda_save[iter-1],sigma_eps_save[iter-1]);
      arma::vec m_beta = compute_post_mean_beta(Bm_beta_stacked[g],y_val[g],Omega,lambda_save[iter-1],sigma_eps_save[iter-1],
                                                beta_0,c_gamma,V_beta);
      
      beta_save_group[g].row(iter) = rmvnorm_1sample(m_beta,V_beta).t();
      beta_current.col(g) = beta_save_group[g].row(iter).t();

      sum_lambda_term += arma::as_scalar( (beta_current.col(g).t() - beta_0.t() )* Omega * (beta_current.col(g) - beta_0));
      arma::mat diff = gamma_current[g].each_col() - gamma_0;
      sum_sigma_gamma_term += arma::accu(arma::square(diff));
    }
    
    
    // Lambda update
    double a_star = a_lambda + (G * p) / 2.0;
    double b_star = b_lambda + 0.5 * sum_lambda_term;
    lambda_save[iter] = rinvgamma_1sample(a_star,b_star);
    
    // Sigma gamma update
    a_star = a_gamma + (k * n_patients) / 2.0;
    b_star = b_gamma + 0.5 * sum_sigma_gamma_term;
    sigma_gamma_save[iter] = rinvgamma_1sample(a_star,b_star);
    
    // Sigma eps update
    a_star = a_eps + (tot_obs) / 2.0;
    b_star = b_eps + 0.5 * sum_sigma_eps_term;
    sigma_eps_save[iter] = rinvgamma_1sample(a_star,b_star);
    
    setTxtProgressBar(bar, iter);
  }
  
  close(bar);
  Rcout << std::endl;
  
  // Acceptance rates 
  List accept_norm = normalize_accepts(accepts,nrun);
  
  // Beta 
  List beta_post =  group_term_post_mean(beta_save_group,nburn);
  
  // Gamma 
  List gamma_post = individual_term_post_mean(gamma_save_group,n_pg,nburn);
  
  // Phi
  List phi_post = individual_term_post_mean(phi_save_group,n_pg,nburn);
  
  // Csi 
  List csi_post = individual_term_post_mean(csi_save_group,n_pg,nburn);
  
  // Xi
  List x_post = individual_term_post_mean(x_save_group,n_pg,nburn);
  
  // Lambda 
  double lambda_post = arma::mean(lambda_save.subvec(nburn, nrun - 1));
  
  // Sigma gamma 
  double sigma_gamma_post = arma::mean(sigma_gamma_save.subvec(nburn, nrun - 1));
  
  // Sigma eps
  double sigma_eps_post = arma::mean(sigma_eps_save.subvec(nburn, nrun - 1));
  
  // sd
  List sd_post = individual_term_post_mean_sd(sd_save_group,n_pg,nburn);
  
  // h
  List h_post = build_h_post(Bm_phi,phi_post,n_pg);
  h_list = build_h_post_cpp(Bm_phi,phi_post,n_pg,y_info["n_obs_max"]);
  
  // Bm
  Bm_beta= build_Bm(h_list,y_info["n_groups"],n_obs,
                    knots_beta,degree_beta,intercept_beta,boundary_beta);
  Bm_gamma = build_Bm(h_list,y_info["n_groups"],n_obs,
                     knots_gamma,degree_gamma,intercept_gamma,boundary_gamma);
  
  // y 
  List y_post = build_y_pred_gardella(Bm_beta,beta_post,Bm_gamma,gamma_post,n_pg);
  std::vector<arma::mat> y_post_mat = pad_list_of_lists(y_post,n_obs,y_info["n_obs_max"]);
  List y_star = build_y_star(h_post,y,n_obs,n_pg); 
  List y_star_smoot = build_y_star(h_post,y_post_mat,n_obs,n_pg);
  
  // Final List 
  List Result; 
  
  // Post
  List post; 
  
  post["beta"]        = beta_post; 
  post["gamma"]       = gamma_post;
  post["phi"]         = phi_post;
  post["csi"]         = csi_post; 
  post["x"]           = x_post;
  post["lambda"]      = lambda_post; 
  post["sigma_gamma"] = sigma_gamma_post; 
  post["sigma_eps"]   = sigma_eps_post; 
  post["zeta"]        = zeta_group;
  post["sd"]          = sd_post;
  
  Result["post"] = post; 
  
  // Full 
  List full; 
  
  Rcpp::List beta_full(beta_save_group.size());
  
  for (size_t g = 0; g < beta_save_group.size(); ++g) {
    beta_full[g] = beta_save_group[g]; // arma::mat → lista di matrici
  }
  
  Rcpp::List gamma_full(gamma_save_group.size());
  Rcpp::List csi_full(gamma_save_group.size());
  Rcpp::List phi_full(gamma_save_group.size());
  Rcpp::List x_full(gamma_save_group.size());
  Rcpp::List sd_full(gamma_save_group.size());
  Rcpp::List y_original(gamma_save_group.size());
  
  for (size_t g = 0; g < gamma_save_group.size(); ++g) {
    Rcpp::List gamma_group(gamma_save_group[g].size());
    Rcpp::List phi_group(gamma_save_group[g].size());
    Rcpp::List csi_group(gamma_save_group[g].size());
    Rcpp::List x_group(gamma_save_group[g].size());
    Rcpp::List sd_group(gamma_save_group[g].size());
    for (size_t i = 0; i < gamma_save_group[g].size(); ++i) {
      gamma_group[i] = gamma_save_group[g][i]; 
      phi_group[i]    = phi_save_group[g][i];
      csi_group[i]    = csi_save_group[g][i];
      x_group[i]      = x_save_group[g][i];
      sd_group[i]     = sd_save_group[g][i];
    }
    gamma_full[g] = gamma_group;
    phi_full[g] = phi_group;
    csi_full[g] = csi_group;
    x_full[g] = x_group;
    sd_full[g] = sd_group;
    y_original[g]      = y[g];
  }
  
  
  full["beta"]        = beta_full; 
  full["gamma"]       = gamma_full;
  full["phi"]         = phi_full;
  full["csi"]         = csi_full; 
  full["x"]           = x_full;
  full["lambda"]      = lambda_save; 
  full["sigma_gamma"] = sigma_gamma_save; 
  full["sigma_eps"]   = sigma_eps_save; 
  full["sd"]          = sd_full; 
  
  Result["full"] = full; 
  
  // Acceptance rates
  Result["acceptance"] = accept_norm; 
  
  // Warping functions 
  Result["h"] = h_post;
  
  // Bm 
  List Bm; 
  
  Bm["Bm_beta"]  = Bm_beta;
  Bm["Bm_gamma"] = Bm_gamma;
  Bm["Bm_phi"]   = Bm_phi;
  
  Result["Bm"] = Bm; 
  
  // y_info 
  Result["y_info"] = y_info; 
  
  // y 
  List y_pred; 
  
  y_pred["y"]            = y_original; 
  y_pred["y_smoot"]      = y_post; 
  y_pred["y_star"]       = y_star; 
  y_pred["y_star_smoot"] = y_star_smoot;
  
  Result["curves"] = y_pred; 
  
  Rcout << "DONE" << std::endl;
  
  return Result;
  
}

List gardella_gaussian(
    std::vector<arma::mat> y,Nullable<int> n_groups,Nullable<IntegerVector> n_per_group,
    int nburn,int niter,int n_knots_beta,int n_knots_gamma,int n_knots_phi,
    Nullable<arma::vec> beta0, Nullable<arma::vec> gamma0,
    double a_eps,double b_eps,double a_lambda,double b_lambda,double a_gamma,double b_gamma, 
    double a_phi, double b_phi, double coeff_var_phi, Nullable<arma::vec> phi0,
    bool INIT_WARP,Nullable<arma::vec> phi_init,
    int degree_beta, int degree_gamma, int degree_phi,
    bool intercept_beta,bool intercept_gamma, bool intercept_phi,
    Nullable<NumericVector> boundary_beta, Nullable<NumericVector> boundary_gamma,Nullable<NumericVector> boundary_phi)
{
  
  // Extract and check the information from y
  List y_info = extract_y_info(y);
  check_params_y(y_info,n_groups,n_per_group);
  
  Rcpp::IntegerVector n_pg = Rcpp::as<Rcpp::IntegerVector>(y_info["n_per_group"]);
  int G = Rcpp::as<int>(y_info["n_groups"]);
  std::vector<Rcpp::IntegerVector> n_obs = y_info["n_obs"];
  std::vector<arma::vec> y_val = y_info["y_val"]; 
  int n_patients = Rcpp::as<int>(y_info["n_patients"]);
  arma::vec n_obs_tot_per_group = y_info["n_obs_tot_per_group"];
  int tot_obs = arma::accu(n_obs_tot_per_group);
  
  // Define the spline parameters
  List beta_spline  = compute_spline_order(degree_beta, intercept_beta, std::string("beta"));
  List gamma_spline = compute_spline_order(degree_gamma, intercept_gamma, std::string("gamma"));
  List phi_spline   = compute_spline_order(degree_phi, intercept_phi, std::string("phi"));
  
  intercept_beta  = beta_spline["intercept"];
  intercept_gamma = gamma_spline["intercept"];
  intercept_phi   = phi_spline["intercept"];
  
  int p = n_knots_beta + Rcpp::as<int>(beta_spline["order_spline"]);
  int k = n_knots_gamma + Rcpp::as<int>(gamma_spline["order_spline"]);
  int q = n_knots_phi + Rcpp::as<int>(phi_spline["order_spline"]);
  
  arma::vec knots_beta  = build_knots(n_knots_beta);
  arma::vec knots_gamma = build_knots(n_knots_gamma);
  arma::vec knots_phi   = build_knots(n_knots_phi);
  
  std::vector<arma::mat> h_list = build_time_matrix(n_obs,y_info["n_obs_max"]);
  
  std::vector<std::vector<arma::mat>> Bm_beta  = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_beta,degree_beta,intercept_beta,boundary_beta);
  std::vector<std::vector<arma::mat>> Bm_gamma = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_gamma,degree_gamma,intercept_gamma,boundary_gamma);
  std::vector<std::vector<arma::mat>> Bm_phi   = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_phi,degree_phi,intercept_phi,boundary_phi);
  
  // Stacked splines
  std::vector<arma::mat> Bm_beta_stacked = stack_Matrix_groups(Bm_beta,p);
  
  // Number of iterations 
  int nrun = nburn + niter;
  
  // Define the parameters 
  
  // Sigma epsilon
  arma::vec sigma_eps_save(nrun);
  sigma_eps_save[0] = rinvgamma_1sample(a_eps,b_eps);
  
  // Lambda 
  arma::vec lambda_save(nrun);
  lambda_save[0] = rinvgamma_1sample(a_lambda,b_lambda);
  
  // Sigma gamma 
  arma::vec sigma_gamma_save(nrun);
  sigma_gamma_save[0] = rinvgamma_1sample(a_gamma,b_gamma);

  // Sigma phi
  arma::vec sigma_phi_save(nrun);
  sigma_phi_save[0] = rinvgamma_1sample(a_phi,b_phi);
  
  // Beta 
  arma::vec beta_0 = check_mean_param(beta0,p,std::string("beta"));
  arma::mat Omega = omega_P_cpp(p);  
  arma::mat Sigma_beta = arma::inv_sympd(Omega / lambda_save[0]);
  
  std::vector<arma::mat> beta_save_group = init_beta_save_group(G,nrun,p,beta_0,Sigma_beta);
  arma::mat beta_current = build_beta_current(beta_save_group,p);
  
  // Gamma 
  arma::vec gamma_0 = check_mean_param(gamma0,k,std::string("gamma"));
  
  arma::mat sigma_gamma_matrix = arma::diagmat(arma::vec(k, arma::fill::ones) * sigma_gamma_save[0]);
  arma::mat inv_sigma_gamma_matrix = arma::inv(sigma_gamma_matrix);
  
  std::vector<std::vector<arma::mat>> gamma_save_group = init_gamma_save_group(G,n_pg,nrun,k,gamma_0,sigma_gamma_matrix);
  std::vector<arma::mat> gamma_current = build_gamma_current(gamma_save_group,n_pg,k);
  
  // Phi 
  arma::vec phi_0 = check_mean_param_gaussian(phi0,q,Rcpp::as<int>(phi_spline["order_spline"]),n_knots_phi,
                                              std::string("phi"));
  
  arma::mat Omega_phi = omega_P_cpp(q);  
  arma::mat Sigma_phi = arma::inv_sympd(Omega_phi / sigma_phi_save[0]);

  std::vector<std::vector<arma::mat>> phi_save_group = init_phi_save_group_gaussian(G,n_pg,nrun,q,phi_0,Sigma_phi,INIT_WARP,phi_init);
  
  // Acceptance rates 
  std::vector<std::vector<arma::vec>> accepts(G);
  
  for (int g = 0; g < G; g++) {
    accepts[g].resize(n_pg[g]);
    for (int i = 0; i < n_pg[g]; i++) {
      accepts[g][i] = arma::zeros(q - 2);
    }
  }

  arma::vec phi_new(q);
  int accettato;

  RObject bar = txtProgressBar(_["min"] = 0, _["max"] = nrun-1, _["style"] = 3);
  
  for (int iter = 1; iter < nrun; iter++) {

    double sum_lambda_term = 0.0; 
    double sum_sigma_gamma_term = 0.0;
    double sum_sigma_phi_term = 0.0;
    double sum_sigma_eps_term = 0.0;
    
    for(int g = 0; g < G; g++){

      arma::uword offset = 0;
      arma::vec c_gamma(y_val[g].n_elem, arma::fill::zeros); 
      
      for(int i = 0; i < n_pg[g]; i++ ){

        const arma::vec y_gi = y[g].col(i).subvec(0, n_obs[g][i] - 1);

        arma::vec beta_current_vec  = beta_save_group[g].row(iter-1).t();
        arma::vec gamma_current_vec = gamma_save_group[g][i].row(iter-1).t();
        
        phi_new  = phi_save_group[g][i].row(iter-1).t();


        for(int j = 1; j < q-1; j++) {
          // MH step
          MH_gardella_gaussian(phi_new,accettato,j,beta_current_vec,gamma_current_vec,
                               sigma_eps_save[iter-1], phi_0, Omega_phi, sigma_phi_save[iter-1],
                               coeff_var_phi,Bm_phi[g][i],y_gi,
                               degree_beta,degree_gamma,intercept_beta,intercept_gamma,
                               boundary_beta,boundary_gamma);

          accepts[g][i][j-1] += accettato;

        }
        
        // Phi update
        phi_save_group[g][i].row(iter) = phi_new.t();
        
        // h update
        arma::vec h_new = Bm_phi[g][i] * phi_new;
        h_list[g].submat(0, i, n_obs[g][i]-1, i) = h_new;
        
        // Bm update 
        Bm_beta[g][i] = build_Bm_individual(h_new,knots_beta,degree_beta,intercept_beta,boundary_beta);
        Bm_gamma[g][i] = build_Bm_individual(h_new,knots_gamma,degree_gamma,intercept_gamma,boundary_gamma); 
        
        // Gamma update
        arma::mat V_gamma_gi = compute_post_cov_gamma(Bm_beta[g][i],beta_current.col(g),Bm_gamma[g][i],
                                                      y[g].col(i).subvec(0, n_obs[g][i] - 1),
                                                      sigma_gamma_save[iter-1],sigma_eps_save[iter-1]);
        arma::vec m_gamma_gi = compute_post_mean_gamma(Bm_beta[g][i],beta_current.col(g),Bm_gamma[g][i],y_gi,
                                                       sigma_gamma_save[iter-1],sigma_eps_save[iter-1],
                                                       gamma_0, V_gamma_gi);
        
        gamma_save_group[g][i].row(iter) = rmvnorm_1sample(m_gamma_gi,V_gamma_gi).t();
        
        gamma_current[g].col(i) = gamma_save_group[g][i].row(iter).t();

        arma::vec contrib_i = Bm_gamma[g][i] * gamma_current[g].col(i);
      
        c_gamma.subvec(offset, offset + contrib_i.n_elem - 1) = contrib_i;
        offset += contrib_i.n_elem;

        sum_sigma_phi_term += arma::as_scalar( (phi_new.t() - phi_0.t() )* Omega_phi * (phi_new - phi_0));
        sum_sigma_eps_term += arma::accu(arma::square( y_gi - Bm_beta[g][i] * beta_current.col(g) - contrib_i ));
        
      }
      
      // Stacked matrix update
      Bm_beta_stacked[g] = stack_single_group(Bm_beta[g],p,n_obs[g]);
      
      // Beta update 
      arma::mat V_beta = compute_post_cov_beta(Bm_beta_stacked[g],Omega,lambda_save[iter-1],sigma_eps_save[iter-1]);
      arma::vec m_beta = compute_post_mean_beta(Bm_beta_stacked[g],y_val[g],Omega,lambda_save[iter-1],sigma_eps_save[iter-1],
                                                beta_0,c_gamma,V_beta);
      
      beta_save_group[g].row(iter) = rmvnorm_1sample(m_beta,V_beta).t();
      beta_current.col(g) = beta_save_group[g].row(iter).t();

      sum_lambda_term += arma::as_scalar( (beta_current.col(g).t() - beta_0.t() )* Omega * (beta_current.col(g) - beta_0));
      arma::mat diff = gamma_current[g].each_col() - gamma_0;
      sum_sigma_gamma_term += arma::accu(arma::square(diff));
    }
    
    
    // Lambda update
    double a_star = a_lambda + (G * p) / 2.0;
    double b_star = b_lambda + 0.5 * sum_lambda_term;
    lambda_save[iter] = rinvgamma_1sample(a_star,b_star);
    
    // Sigma gamma update
    a_star = a_gamma + (k * n_patients) / 2.0;
    b_star = b_gamma + 0.5 * sum_sigma_gamma_term;
    sigma_gamma_save[iter] = rinvgamma_1sample(a_star,b_star);

    // Sigma phi update
    a_star = a_phi + (q * G) / 2.0;
    b_star = b_phi + 0.5 * sum_sigma_phi_term;
    sigma_phi_save[iter] = rinvgamma_1sample(a_star,b_star);
    
    // Sigma eps update
    a_star = a_eps + (tot_obs) / 2.0;
    b_star = b_eps + 0.5 * sum_sigma_eps_term;
    sigma_eps_save[iter] = rinvgamma_1sample(a_star,b_star);
    
    setTxtProgressBar(bar, iter);
  }
  
  close(bar);
  Rcout << std::endl;

  // Acceptance rates 
  List accept_norm(G);
  
  for(int g = 0; g < G; g++){
    List accept_norm_group(n_pg[g]);
    for(int i = 0; i < n_pg[g]; i++ ){
      accept_norm_group[i] = accepts[g][i] / nrun;
    }
    accept_norm[g] = accept_norm_group;
  }
  
  
  // Beta 
  List beta_post =  group_term_post_mean(beta_save_group,nburn);
  
  // Gamma 
  List gamma_post = individual_term_post_mean(gamma_save_group,n_pg,nburn);
  
  // Phi
  List phi_post = individual_term_post_mean(phi_save_group,n_pg,nburn);
  
  // Lambda 
  double lambda_post = arma::mean(lambda_save.subvec(nburn, nrun - 1));
  
  // Sigma gamma 
  double sigma_gamma_post = arma::mean(sigma_gamma_save.subvec(nburn, nrun - 1));

  // Sigma phi
  double sigma_phi_post = arma::mean(sigma_phi_save.subvec(nburn, nrun - 1));
  
  // Sigma eps
  double sigma_eps_post = arma::mean(sigma_eps_save.subvec(nburn, nrun - 1));
  
  // h
  List h_post = build_h_post(Bm_phi,phi_post,n_pg);
  h_list = build_h_post_cpp(Bm_phi,phi_post,n_pg,y_info["n_obs_max"]);
  
  // Bm
  Bm_beta= build_Bm(h_list,y_info["n_groups"],n_obs,
                    knots_beta,degree_beta,intercept_beta,boundary_beta);
  Bm_gamma = build_Bm(h_list,y_info["n_groups"],n_obs,
                     knots_gamma,degree_gamma,intercept_gamma,boundary_gamma);
  
  // y 
  List y_post = build_y_pred_gardella(Bm_beta,beta_post,Bm_gamma,gamma_post,n_pg);
  std::vector<arma::mat> y_post_mat = pad_list_of_lists(y_post,n_obs,y_info["n_obs_max"]);
  List y_star = build_y_star(h_post,y,n_obs,n_pg); 
  List y_star_smoot = build_y_star(h_post,y_post_mat,n_obs,n_pg);
  
  // Final List 
  List Result; 
  
  // Post
  List post; 
  
  post["beta"]        = beta_post; 
  post["gamma"]       = gamma_post;
  post["phi"]         = phi_post;
  post["lambda"]      = lambda_post; 
  post["sigma_gamma"] = sigma_gamma_post;
  post["sigma_phi"]   = sigma_phi_post;
  post["sigma_eps"]   = sigma_eps_post; 
  
  Result["post"] = post; 
  
  // Full 
  List full; 
  
  Rcpp::List beta_full(beta_save_group.size());
  
  for (size_t g = 0; g < beta_save_group.size(); ++g) {
    beta_full[g] = beta_save_group[g]; // arma::mat → lista di matrici
  }
  
  Rcpp::List gamma_full(gamma_save_group.size());
  Rcpp::List phi_full(gamma_save_group.size());
  Rcpp::List y_original(gamma_save_group.size());
  
  for (size_t g = 0; g < gamma_save_group.size(); ++g) {
    Rcpp::List gamma_group(gamma_save_group[g].size());
    Rcpp::List phi_group(gamma_save_group[g].size());
    for (size_t i = 0; i < gamma_save_group[g].size(); ++i) {
      gamma_group[i] = gamma_save_group[g][i]; 
      phi_group[i]    = phi_save_group[g][i];
    }
    gamma_full[g] = gamma_group;
    phi_full[g] = phi_group;
    y_original[g]      = y[g];
  }
  
  
  full["beta"]        = beta_full; 
  full["gamma"]       = gamma_full;
  full["phi"]         = phi_full;
  full["lambda"]      = lambda_save; 
  full["sigma_gamma"] = sigma_gamma_save; 
  full["sigma_phi"]   = sigma_phi_save; 
  full["sigma_eps"]   = sigma_eps_save; 
  
  Result["full"] = full; 
  
  // Acceptance rates
  Result["acceptance"] = accept_norm; 
  
  // Warping functions 
  Result["h"] = h_post;
  
  // Bm 
  List Bm; 
  
  Bm["Bm_beta"]  = Bm_beta;
  Bm["Bm_gamma"] = Bm_gamma;
  Bm["Bm_phi"]   = Bm_phi;
  
  Result["Bm"] = Bm; 
  
  // y_info 
  Result["y_info"] = y_info; 
  
  // y 
  List y_pred; 
  
  y_pred["y"]            = y_original; 
  y_pred["y_smoot"]      = y_post; 
  y_pred["y_star"]       = y_star; 
  y_pred["y_star_smoot"] = y_star_smoot;
  
  Result["curves"] = y_pred; 
  
  Rcout << "DONE" << std::endl;
  
  return Result;
  
}

List telesca_gaussian(
    std::vector<arma::mat> y,Nullable<int> n_groups,Nullable<IntegerVector> n_per_group,
    int nburn,int niter,int n_knots_beta,int n_knots_phi,
    Nullable<arma::vec> beta0, double m_c0, double sigma_c0, double m_a0, double sigma_a0,
    double a_c, double b_c, double a_a, double b_a,
    double a_eps,double b_eps,double a_lambda,double b_lambda, 
    double a_phi, double b_phi, double coeff_var_phi, Nullable<arma::vec> phi0,
    bool INIT_WARP,Nullable<arma::vec> phi_init,
    int degree_beta, int degree_phi,
    bool intercept_beta, bool intercept_phi,
    Nullable<NumericVector> boundary_beta,Nullable<NumericVector> boundary_phi)
{
  
  // Extract and check the information from y
  List y_info = extract_y_info(y);
  check_params_y(y_info,n_groups,n_per_group);
  
  int G = Rcpp::as<int>(y_info["n_groups"]);

  if(G > 1){
    Rcout<<"More than 1 group detected. The function will run separately for each group."<<std::endl;
    Rcout << std::endl;
  }

  Rcpp::IntegerVector n_pg = Rcpp::as<Rcpp::IntegerVector>(y_info["n_per_group"]);
  std::vector<Rcpp::IntegerVector> n_obs = y_info["n_obs"];
  std::vector<arma::vec> y_val = y_info["y_val"]; 
  arma::vec n_obs_tot_per_group = y_info["n_obs_tot_per_group"];
  
  // Define the spline parameters
  List beta_spline  = compute_spline_order(degree_beta, intercept_beta, std::string("beta"));
  List phi_spline   = compute_spline_order(degree_phi, intercept_phi, std::string("phi"));
  
  intercept_beta  = beta_spline["intercept"];
  intercept_phi   = phi_spline["intercept"];
  
  int p = n_knots_beta + Rcpp::as<int>(beta_spline["order_spline"]);
  int q = n_knots_phi + Rcpp::as<int>(phi_spline["order_spline"]);
  
  arma::vec knots_beta  = build_knots(n_knots_beta);
  arma::vec knots_phi   = build_knots(n_knots_phi);
  
  std::vector<arma::mat> h_list = build_time_matrix(n_obs,y_info["n_obs_max"]);
  
  std::vector<std::vector<arma::mat>> Bm_beta  = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_beta,degree_beta,intercept_beta,boundary_beta);
  std::vector<std::vector<arma::mat>> Bm_phi   = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_phi,degree_phi,intercept_phi,boundary_phi);
  
  // Number of iterations 
  int nrun = nburn + niter;

  List Result;
  
  // Loop over the groups 

  for(int g = 0; g < G; g++){

    if(G > 1){
      Rcout<<"Running group "<<g+1<<std::endl;
      Rcout << std::endl;
    }

    // Define the parameters 
  
    // Sigma epsilon
    arma::vec sigma_eps_save(nrun);
    sigma_eps_save[0] = rinvgamma_1sample(a_eps,b_eps);
  
    // Lambda 
    arma::vec lambda_save(nrun);
    lambda_save[0] = rinvgamma_1sample(a_lambda,b_lambda);
    
    // Sigma a
    arma::vec sigma_a_save(nrun);
    sigma_a_save[0] = rinvgamma_1sample(a_a,b_a);

    // Sigma c  
    arma::vec sigma_c_save(nrun);
    sigma_c_save[0] = rinvgamma_1sample(a_c,b_c);

    // a0
    arma::vec a0_save(nrun);
    a0_save[0] = rnorm_1sample(m_a0, std::sqrt(sigma_a0));

    // c0
    arma::vec c0_save(nrun);
    c0_save[0] = rnorm_1sample(m_c0, std::sqrt(sigma_c0));

    // a
    arma::mat a_save = init_a_c_save(nrun,n_pg[g],a0_save[0],sigma_a_save[0]);

    // c
    arma::mat c_save = init_a_c_save(nrun,n_pg[g],c0_save[0],sigma_c_save[0]);   

    // Sigma phi
    arma::vec sigma_phi_save(nrun);
    sigma_phi_save[0] = rinvgamma_1sample(a_phi,b_phi);
    
    // Beta 
    arma::vec beta_0 = check_mean_param(beta0,p,std::string("beta"));
    arma::mat Omega = omega_P_cpp(p);  
    arma::mat Sigma_beta = arma::inv_sympd(Omega / lambda_save[0]);
    
    arma::mat beta_save = init_beta_save(nrun,p,beta_0,Sigma_beta);
    
    // Phi 
    arma::vec phi_0 = check_mean_param_gaussian(phi0,q,Rcpp::as<int>(phi_spline["order_spline"]),n_knots_phi,
                                                std::string("phi"));
    
    arma::mat Omega_phi = omega_P_cpp(q);  
    arma::mat Sigma_phi = arma::inv_sympd(Omega_phi / sigma_phi_save[0]);

    std::vector<arma::mat> phi_save = init_phi_save_gaussian(n_pg[g],nrun,q,phi_0,Sigma_phi,INIT_WARP,phi_init);
    
    // Stacked splines
    arma::mat Bm_beta_stacked = stack_single_group_telesca(Bm_beta[g],a_save.row(0),p,n_obs[g]);
    
    // Acceptance rates 
    std::vector<arma::vec>accepts(n_pg[g]);
    
    for (int i = 0; i < n_pg[g]; i++) {
        accepts[i] = arma::zeros(q - 2);
    }
    
    arma::vec phi_new(q);
    int accettato;

    arma::mat y_g = y[g];
    IntegerVector n_obs_g = n_obs[g];
    std::vector<arma::mat> Bm_beta_g = Bm_beta[g];
    std::vector<arma::mat> Bm_phi_g = Bm_phi[g]; 
    arma::vec y_val_g = y_val[g];
    int tot_obs_g = n_obs_tot_per_group[g];

    arma::vec c_stacked(n_obs_tot_per_group[g], arma::fill::zeros);

    RObject bar = txtProgressBar(_["min"] = 0, _["max"] = nrun-1, _["style"] = 3);
  
    for (int iter = 1; iter < nrun; iter++) {

      int row_offset_stacked = 0;

      double sum_lambda_term = 0.0; 
      double sum_sigma_phi_term = 0.0;
      double sum_sigma_eps_term = 0.0; 

      arma::vec beta_current_vec  = beta_save.row(iter-1).t();

      arma::vec d = {sigma_a_save[iter-1], sigma_c_save[iter-1]};
      arma::mat inv_Sigma_a_c = arma::diagmat(d);
      arma::vec vec_c0_a0 = {c0_save[iter-1], a0_save[iter-1]};
        
      for(int i = 0; i < n_pg[g]; i++ ){

        const arma::vec y_gi = y_g.col(i).subvec(0, n_obs_g[i] - 1);
                  
        phi_new  = phi_save[i].row(iter-1).t();


        for(int j = 1; j < q-1; j++) {
          // MH step
          MH_telesca_gaussian(phi_new,accettato,j,beta_current_vec,a_save(iter,i),c_save(iter,i),
                              sigma_eps_save[iter-1], phi_0, Omega_phi, sigma_phi_save[iter-1],
                              coeff_var_phi,Bm_phi_g[i],y_gi,
                              degree_beta,intercept_beta,boundary_beta);

          accepts[i][j-1] += accettato;

        }
          
        // Phi update
        phi_save[i].row(iter) = phi_new.t();
        
        // h update
        arma::vec h_new = Bm_phi_g[i] * phi_new;
        h_list[g].submat(0, i, n_obs_g[i]-1, i) = h_new;

        // a_i, c_i update 
        arma::mat V_ac = compute_post_cov_a_c(n_obs_g[i],Bm_beta_g[i],beta_current_vec,inv_Sigma_a_c,sigma_eps_save[iter-1]);
        arma::vec m_ac = compute_post_mean_a_c(n_obs_g[i],Bm_beta_g[i],beta_current_vec,inv_Sigma_a_c,sigma_eps_save[iter-1],
                                                y_gi,vec_c0_a0,V_ac);
        arma::vec ac_sample = rmvnorm_1sample(m_ac,V_ac);
        c_save(iter, i) = ac_sample[0];
        a_save(iter, i) = ac_sample[1];
        
        // Bm update 
        Bm_beta_g[i] = build_Bm_individual(h_new,knots_beta,degree_beta,intercept_beta,boundary_beta); 
        
        // Stacked matrix update
        Bm_beta_stacked.rows(row_offset_stacked, row_offset_stacked + Bm_beta_g[i].n_rows - 1) = Bm_beta_g[i] * a_save(iter-1,i);
        row_offset_stacked += Bm_beta_g[i].n_rows;

        double c_i = c_save(iter-1, i);
        c_stacked.subvec(0, n_obs_g[i] - 1).fill(c_i);
        arma::vec contrib_i = c_stacked.subvec(0, n_obs_g[i] - 1);

        sum_sigma_phi_term += arma::as_scalar( (phi_new.t() - phi_0.t() )* Omega_phi * (phi_new - phi_0));
        sum_sigma_eps_term += arma::accu(arma::square( y_gi - Bm_beta_g[i] * beta_current_vec - contrib_i ));
          
      }
        
      // Beta update 
      arma::mat V_beta = compute_post_cov_beta(Bm_beta_stacked,Omega,lambda_save[iter-1],sigma_eps_save[iter-1]);
      arma::vec m_beta = compute_post_mean_beta(Bm_beta_stacked,y_val_g,Omega,lambda_save[iter-1],sigma_eps_save[iter-1],
                                                        beta_0,c_stacked,V_beta);
      
      beta_save.row(iter) = rmvnorm_1sample(m_beta,V_beta).t();

      sum_lambda_term += arma::as_scalar( (beta_save.row(iter) - beta_0.t() )* Omega * (beta_save.row(iter).t() - beta_0));

      // Lambda update
      double a_star = a_lambda + (G * p) / 2.0;
      double b_star = b_lambda + 0.5 * sum_lambda_term;
      lambda_save[iter] = rinvgamma_1sample(a_star,b_star);

      // a0 update
      double sigma_a0_star = 1.0 / (1.0 / sigma_a0 + n_pg[g] / sigma_a_save[iter-1]);
      double a0_star = sigma_a0_star * (m_a0 / sigma_a0 + arma::accu(a_save.row(iter-1)) / sigma_a_save[iter-1]);
      a0_save[iter] = rnorm_1sample(a0_star, std::sqrt(sigma_a0_star));

      // c0 update
      double sigma_c0_star = 1.0 / (1.0 / sigma_c0 + n_pg[g] / sigma_c_save[iter-1]);
      double c0_star = sigma_c0_star * (m_c0 / sigma_c0 + arma::accu(c_save.row(iter-1)) / sigma_c_save[iter-1]);
      c0_save[iter] = rnorm_1sample(c0_star, std::sqrt(sigma_c0_star));

      // Sigma a update
      a_star = a_a + (n_pg[g]) / 2.0;
      b_star = b_a + 0.5 * arma::accu(arma::square(a_save.row(iter) - a0_save[iter]));
      sigma_a_save[iter] = rinvgamma_1sample(a_star,b_star);

      // Sigma c update
      a_star = a_c + (n_pg[g]) / 2.0;
      b_star = b_c + 0.5 * arma::accu(arma::square(c_save.row(iter) - c0_save[iter]));
      sigma_c_save[iter] = rinvgamma_1sample(a_star,b_star);

      // Sigma phi update
      a_star = a_phi + (q * G) / 2.0;
      b_star = b_phi + 0.5 * sum_sigma_phi_term;
      sigma_phi_save[iter] = rinvgamma_1sample(a_star,b_star);
      
      // Sigma eps update
      a_star = a_eps + (tot_obs_g) / 2.0;
      b_star = b_eps + 0.5 * sum_sigma_eps_term;
      sigma_eps_save[iter] = rinvgamma_1sample(a_star,b_star);
      
      setTxtProgressBar(bar, iter);
    }
    
    close(bar);
    Rcout << std::endl;

    // Save results for group g

    // Acceptance rates 
    List accept_norm(n_pg[g]);
    for(int i = 0; i < n_pg[g]; i++ ){
        accept_norm[i] = accepts[i] / (nrun - 1);
    }

    // Beta 
    arma::mat beta_post_mat = beta_save.rows(nburn, nrun - 1);
    arma::vec beta_post = arma::mean(beta_post_mat, 0).t();

    // a 
    List a_post(n_pg[g]); 
    
    arma::mat a_post_mat = a_save.rows(nburn, a_save.n_rows - 1);
    arma::rowvec a_mean = arma::mean(a_post_mat, 0);
    
    for(int i = 0; i < n_pg[g]; i++){
      a_post[i] = a_mean[i]; 
    }
    
    // c 
    List c_post(n_pg[g]);
    arma::mat c_post_mat = c_save.rows(nburn, c_save.n_rows - 1);
    arma::rowvec c_mean = arma::mean(c_post_mat, 0);
    
    for(int i = 0; i < n_pg[g]; i++){
      c_post[i] = c_mean[i]; 
    }
    
    // Phi
    List phi_post = group_term_post_mean(phi_save,nburn);

    // a0 
    double a0_post = arma::mean(a0_save.subvec(nburn, nrun - 1));

    // c0 
    double c0_post = arma::mean(c0_save.subvec(nburn, nrun - 1));
    
    // Lambda 
    double lambda_post = arma::mean(lambda_save.subvec(nburn, nrun - 1));
    
    // Sigma a 
    double sigma_a_post = arma::mean(sigma_a_save.subvec(nburn, nrun - 1));

    // Sigma c 
    double sigma_c_post = arma::mean(sigma_c_save.subvec(nburn, nrun - 1));

    // Sigma phi
    double sigma_phi_post = arma::mean(sigma_phi_save.subvec(nburn, nrun - 1));
    
    // Sigma eps
    double sigma_eps_post = arma::mean(sigma_eps_save.subvec(nburn, nrun - 1));
    
    // h
    List h_post = build_h_post_single_group(Bm_phi[g],phi_post,n_pg[g]);
    arma::mat h_mat = build_h_post_cpp_single_group(Bm_phi[g],phi_post,n_pg[g],y_info["n_obs_max"]);
    
    // Bm
    std::vector<arma::mat> Bm_beta_post = build_Bm_single_group(h_mat,n_obs[g],
                            knots_beta,degree_beta,intercept_beta,boundary_beta);

    // y 
    Rcpp::List y_post = build_y_pred_telesca(Bm_beta_post,beta_post,a_post,c_post,n_pg[g]);
    arma::mat y_post_mat = pad_list_of_lists_single_group(y_post,n_obs[g],y_info["n_obs_max"]);
    List y_star = build_y_star_single_group(h_post,y[g],n_obs[g],n_pg[g]); 
    List y_star_smoot = build_y_star_single_group(h_post,y_post_mat,n_obs[g],n_pg[g]);
    
    // Final List 
    List Result_g; 
    
    // Post
    List post; 
    
    post["beta"]        = beta_post; 
    post["a"]           = a_post;
    post["c"]           = c_post;
    post["phi"]         = phi_post;
    post["a0"]          = a0_post;
    post["c0"]          = c0_post;
    post["lambda"]      = lambda_post; 
    post["sigma_a"]     = sigma_a_post;
    post["sigma_c"]     = sigma_c_post;
    post["sigma_phi"]   = sigma_phi_post;
    post["sigma_eps"]   = sigma_eps_post; 
    
    Result_g["post"] = post; 
    
    // Full 
    List full; 
    
    Rcpp::List beta_full(1);
    beta_full[0] = beta_save;
    
    Rcpp::List y_original(1);
    y_original[0] = y[g];
  
    Rcpp::List phi_full(n_pg[g]);
    Rcpp::List a_full(n_pg[g]);
    Rcpp::List c_full(n_pg[g]);
  
    for (size_t i = 0; i < phi_save.size(); ++i) {
        phi_full[i]    = phi_save[i];
        a_full[i]      = a_save.col(i);
        c_full[i]      = c_save.col(i);
    }
    
    full["beta"]        = beta_full[0]; 
    full["a"]           = a_full;
    full["c"]           = c_full;
    full["phi"]         = phi_full;
    full["a0"]          = a0_save;
    full["c0"]          = c0_save;
    full["lambda"]      = lambda_save; 
    full["sigma_a"]     = sigma_a_save;
    full["sigma_c"]     = sigma_c_save;
    full["sigma_phi"]   = sigma_phi_save; 
    full["sigma_eps"]   = sigma_eps_save; 
    
    Result_g["full"] = full; 
    
    // Acceptance rates
    Result_g["acceptance"] = accept_norm; 
    
    // Warping functions 
    Result_g["h"] = h_post;
    
    // Bm 
    List Bm; 
    
    Bm["Bm_beta"]  = Bm_beta_post;
    Bm["Bm_phi"]   = Bm_phi[g];
    
    Result_g["Bm"] = Bm; 
    
    // y_info 
    List y_info_group_g; 

    y_info_group_g["n_groups"] = 1;
    y_info_group_g["n_patients"] = n_pg[g];
    y_info_group_g["n_obs"] = n_obs[g];
    y_info_group_g["n_obs_max"] = n_obs_tot_per_group[g];
    y_info_group_g["y_val"] = y_val[g];

    Result_g["y_info"] = y_info_group_g;
    
    // y 
    List y_pred; 
    
    y_pred["y"]            = y_original; 
    y_pred["y_smoot"]      = y_post; 
    y_pred["y_star"]       = y_star; 
    y_pred["y_star_smoot"] = y_star_smoot;
    
    Result_g["curves"] = y_pred; 
    
    if(G > 1){

      Rcout << "DONE group " <<g+1 << std::endl;
      Rcout << std::endl;
      Result["group_" + std::to_string(g + 1)] = Result_g;

    } else {
      Rcout << "DONE" << std::endl;
      return Result_g; 
    }

  }
  
  Rcout << "DONE" << std::endl;

  return Result;
}

List telesca_gamma_adaptation(
    std::vector<arma::mat> y,Nullable<int> n_groups,Nullable<IntegerVector> n_per_group,
    int nburn,int niter,int n_knots_beta,int n_knots_phi,
    Nullable<arma::vec> beta0, double m_c0, double sigma_c0, double m_a0, double sigma_a0,
    double a_c, double b_c, double a_a, double b_a,
    double a_eps,double b_eps,double a_lambda,double b_lambda, 
    Nullable<arma::vec> a_f, double b_f,
    Nullable<arma::mat> zeta_0, double zeta_0_coeff,
    Nullable<double> sd0,double sd_min,double sd_max,double target_alpha,double LAMB,
    double EPS,int n0_zeta,bool INIT_WARP,Nullable<arma::vec> phi_init,double SOMMA_csi,
    int degree_beta, int degree_phi,
    bool intercept_beta, bool intercept_phi,
    Nullable<NumericVector> boundary_beta,Nullable<NumericVector> boundary_phi)
{
  
  // Check sd_min < sd_max 
  check_rho_clipping(sd_min,sd_max);

  // Extract and check the information from y
  List y_info = extract_y_info(y);
  check_params_y(y_info,n_groups,n_per_group);
  
  int G = Rcpp::as<int>(y_info["n_groups"]);

  if(G > 1){
    Rcout<<"More than 1 group detected. The function will run separately for each group."<<std::endl;
    Rcout << std::endl;
  }

  Rcpp::IntegerVector n_pg = Rcpp::as<Rcpp::IntegerVector>(y_info["n_per_group"]);
  std::vector<Rcpp::IntegerVector> n_obs = y_info["n_obs"];
  std::vector<arma::vec> y_val = y_info["y_val"]; 
  arma::vec n_obs_tot_per_group = y_info["n_obs_tot_per_group"];
  
  // Define the spline parameters
  List beta_spline  = compute_spline_order(degree_beta, intercept_beta, std::string("beta"));
  List phi_spline   = compute_spline_order(degree_phi, intercept_phi, std::string("phi"));
  
  intercept_beta  = beta_spline["intercept"];
  intercept_phi   = phi_spline["intercept"];
  
  int p = n_knots_beta + Rcpp::as<int>(beta_spline["order_spline"]);
  int q = n_knots_phi + Rcpp::as<int>(phi_spline["order_spline"]);

  // Check a_f 
  arma::vec a_f_vec = check_af(a_f,q);
  
  arma::vec knots_beta  = build_knots(n_knots_beta);
  arma::vec knots_phi   = build_knots(n_knots_phi);
  
  std::vector<arma::mat> h_list = build_time_matrix(n_obs,y_info["n_obs_max"]);
  
  std::vector<std::vector<arma::mat>> Bm_beta  = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_beta,degree_beta,intercept_beta,boundary_beta);
  std::vector<std::vector<arma::mat>> Bm_phi   = build_Bm(h_list,y_info["n_groups"],n_obs,
                                                          knots_phi,degree_phi,intercept_phi,boundary_phi);
  
  // Number of iterations 
  int nrun = nburn + niter;

  List Result;
  
  // Loop over the groups 

  for(int g = 0; g < G; g++){

    if(G > 1){
      Rcout<<"Running group "<<g+1<<std::endl;
      Rcout << std::endl;
    }

    // Define the parameters 
  
    // Sigma epsilon
    arma::vec sigma_eps_save(nrun);
    sigma_eps_save[0] = rinvgamma_1sample(a_eps,b_eps);
  
    // Lambda 
    arma::vec lambda_save(nrun);
    lambda_save[0] = rinvgamma_1sample(a_lambda,b_lambda);
    
    // Sigma a
    arma::vec sigma_a_save(nrun);
    sigma_a_save[0] = rinvgamma_1sample(a_a,b_a);

    // Sigma c  
    arma::vec sigma_c_save(nrun);
    sigma_c_save[0] = rinvgamma_1sample(a_c,b_c);

    // a0
    arma::vec a0_save(nrun);
    a0_save[0] = rnorm_1sample(m_a0, std::sqrt(sigma_a0));

    // c0
    arma::vec c0_save(nrun);
    c0_save[0] = rnorm_1sample(m_c0, std::sqrt(sigma_c0));

    // a
    arma::mat a_save = init_a_c_save(nrun,n_pg[g],a0_save[0],sigma_a_save[0]);

    // c
    arma::mat c_save = init_a_c_save(nrun,n_pg[g],c0_save[0],sigma_c_save[0]);   
    
    // Beta 
    arma::vec beta_0 = check_mean_param(beta0,p,std::string("beta"));
    arma::mat Omega = omega_P_cpp(p);  
    arma::mat Sigma_beta = arma::inv_sympd(Omega / lambda_save[0]);
    
    arma::mat beta_save = init_beta_save(nrun,p,beta_0,Sigma_beta);
    
    // Csi 
    std::vector<arma::mat> csi_save = init_csi_save(n_pg[g],nrun,q,INIT_WARP,phi_init,SOMMA_csi,a_f_vec,b_f);

    // Phi 
    std::vector<arma::mat> phi_save = init_phi_save_gamma_adapt(n_pg[g],nrun,csi_save);
    
    // x = log(csi)
    std::vector<arma::mat> x_save = init_x_save(n_pg[g],nrun,csi_save);
    
    // MH parameters 
    // Zeta 
    std::vector<arma::mat> zeta = init_zeta(n_pg[g],q,zeta_0_coeff,zeta_0);
    
    // sd 
    std::vector<arma::vec> sd_save = init_sd_save(n_pg[g],nrun,q,sd0,sd_min,sd_max);
  
    // Acceptance rates 
    Rcpp::IntegerVector accepts = Rcpp::IntegerVector(n_pg[g], 0);

    arma::vec x_new(q);
    int accettato;
    double alpha;
    
    // Stacked splines
    arma::mat Bm_beta_stacked = stack_single_group_telesca(Bm_beta[g],a_save.row(0),p,n_obs[g]);

    arma::mat y_g = y[g];
    IntegerVector n_obs_g = n_obs[g];
    std::vector<arma::mat> Bm_beta_g = Bm_beta[g];
    std::vector<arma::mat> Bm_phi_g = Bm_phi[g]; 
    arma::vec y_val_g = y_val[g];
    int tot_obs_g = n_obs_tot_per_group[g];

    arma::vec c_stacked(n_obs_tot_per_group[g], arma::fill::zeros);

    RObject bar = txtProgressBar(_["min"] = 0, _["max"] = nrun-1, _["style"] = 3);
  
    for (int iter = 1; iter < nrun; iter++) {

      // w update 
      double w = std::pow(iter, -LAMB);

      int row_offset_stacked = 0;

      double sum_lambda_term = 0.0; 
      double sum_sigma_eps_term = 0.0; 

      arma::vec beta_current_vec  = beta_save.row(iter-1).t();

      arma::vec d = {sigma_a_save[iter-1], sigma_c_save[iter-1]};
      arma::mat inv_Sigma_a_c = arma::diagmat(d);
      arma::vec vec_c0_a0 = {c0_save[iter-1], a0_save[iter-1]};
        
      for(int i = 0; i < n_pg[g]; i++ ){

        const arma::vec y_gi = y_g.col(i).subvec(0, n_obs_g[i] - 1);
                  
        x_new  = x_save[i].row(iter-1).t();
        
        // MH step
        MH_telesca_gamma_adapt(x_new,alpha,accettato,x_save[i].row(iter-1).t(),zeta[i],
                                beta_save.row(iter-1).t(),a_save(iter-1,i),c_save(iter-1,i),
                                sigma_eps_save[iter-1], a_f_vec, b_f,Bm_phi_g[i],y_gi,
                                degree_beta,intercept_beta,boundary_beta); 
        
        // Update acceptance rates 
        accepts[i] += accettato;
      
        x_save[i].row(iter) = x_new.t();
        
        // sd update;
        sd_save[i][iter] = rho_clipping(sd_save[i][iter-1] + w * (alpha - target_alpha),sd_min,sd_max);
        
        // zeta update 
        if (iter > n0_zeta)
          {  
             zeta[i] = update_zeta(x_save[i],sd_save[i][iter],iter,q,EPS);
          }
      
        // Csi update
        csi_save[i].row(iter) = build_csi_from_x(x_save[i].row(iter).t(),q).t(); 
        
        // Phi update
        phi_save[i].row(iter) = build_phi_from_csi(csi_save[i].row(iter).t(),q).t();
        
        // h update
        arma::vec h_new = Bm_phi_g[i] * phi_save[i].row(iter).t();
        h_list[g].submat(0, i, n_obs_g[i]-1, i) = h_new;

        // a_i, c_i update 
        arma::mat V_ac = compute_post_cov_a_c(n_obs_g[i],Bm_beta_g[i],beta_current_vec,inv_Sigma_a_c,sigma_eps_save[iter-1]);
        arma::vec m_ac = compute_post_mean_a_c(n_obs_g[i],Bm_beta_g[i],beta_current_vec,inv_Sigma_a_c,sigma_eps_save[iter-1],
                                                y_gi,vec_c0_a0,V_ac);
        arma::vec ac_sample = rmvnorm_1sample(m_ac,V_ac);
        c_save(iter, i) = ac_sample[0];
        a_save(iter, i) = ac_sample[1];
        
        // Bm update 
        Bm_beta_g[i] = build_Bm_individual(h_new,knots_beta,degree_beta,intercept_beta,boundary_beta); 
        
        // Stacked matrix update
        Bm_beta_stacked.rows(row_offset_stacked, row_offset_stacked + Bm_beta_g[i].n_rows - 1) = Bm_beta_g[i] * a_save(iter-1,i);
        row_offset_stacked += Bm_beta_g[i].n_rows;

        double c_i = c_save(iter-1, i);
        c_stacked.subvec(0, n_obs_g[i] - 1).fill(c_i);
        arma::vec contrib_i = c_stacked.subvec(0, n_obs_g[i] - 1);

        sum_sigma_eps_term += arma::accu(arma::square( y_gi - Bm_beta_g[i] * beta_current_vec - contrib_i ));
          
      }      
        
      // Beta update 
      arma::mat V_beta = compute_post_cov_beta(Bm_beta_stacked,Omega,lambda_save[iter-1],sigma_eps_save[iter-1]);
      arma::vec m_beta = compute_post_mean_beta(Bm_beta_stacked,y_val_g,Omega,lambda_save[iter-1],sigma_eps_save[iter-1],
                                                        beta_0,c_stacked,V_beta);
      
      beta_save.row(iter) = rmvnorm_1sample(m_beta,V_beta).t();

      sum_lambda_term += arma::as_scalar( (beta_save.row(iter) - beta_0.t() )* Omega * (beta_save.row(iter).t() - beta_0));

      // Lambda update
      double a_star = a_lambda + (G * p) / 2.0;
      double b_star = b_lambda + 0.5 * sum_lambda_term;
      lambda_save[iter] = rinvgamma_1sample(a_star,b_star);

      // a0 update
      double sigma_a0_star = 1.0 / (1.0 / sigma_a0 + n_pg[g] / sigma_a_save[iter-1]);
      double a0_star = sigma_a0_star * (m_a0 / sigma_a0 + arma::accu(a_save.row(iter-1)) / sigma_a_save[iter-1]);
      a0_save[iter] = rnorm_1sample(a0_star, std::sqrt(sigma_a0_star));

      // c0 update
      double sigma_c0_star = 1.0 / (1.0 / sigma_c0 + n_pg[g] / sigma_c_save[iter-1]);
      double c0_star = sigma_c0_star * (m_c0 / sigma_c0 + arma::accu(c_save.row(iter-1)) / sigma_c_save[iter-1]);
      c0_save[iter] = rnorm_1sample(c0_star, std::sqrt(sigma_c0_star));

      // Sigma a update
      a_star = a_a + (n_pg[g]) / 2.0;
      b_star = b_a + 0.5 * arma::accu(arma::square(a_save.row(iter) - a0_save[iter]));
      sigma_a_save[iter] = rinvgamma_1sample(a_star,b_star);

      // Sigma c update
      a_star = a_c + (n_pg[g]) / 2.0;
      b_star = b_c + 0.5 * arma::accu(arma::square(c_save.row(iter) - c0_save[iter]));
      sigma_c_save[iter] = rinvgamma_1sample(a_star,b_star);
      
      // Sigma eps update
      a_star = a_eps + (tot_obs_g) / 2.0;
      b_star = b_eps + 0.5 * sum_sigma_eps_term;
      sigma_eps_save[iter] = rinvgamma_1sample(a_star,b_star);
      
      setTxtProgressBar(bar, iter);
    }
    
    close(bar);
    Rcout << std::endl;

    // Save results for group g 

    // Acceptance rates 
    List accept_norm(n_pg[g]);
    for(int i = 0; i < n_pg[g]; i++ ){
       accept_norm[i] = static_cast<double>(accepts[i]) / (nrun-1);
    }

    // Beta 
    arma::mat beta_post_mat = beta_save.rows(nburn, nrun - 1);
    arma::vec beta_post = arma::mean(beta_post_mat, 0).t();

    // a 
    List a_post(n_pg[g]); 
    
    arma::mat a_post_mat = a_save.rows(nburn, a_save.n_rows - 1);
    arma::rowvec a_mean = arma::mean(a_post_mat, 0);
    
    for(int i = 0; i < n_pg[g]; i++){
      a_post[i] = a_mean[i]; 
    }
    
    // c 
    List c_post(n_pg[g]);
    arma::mat c_post_mat = c_save.rows(nburn, c_save.n_rows - 1);
    arma::rowvec c_mean = arma::mean(c_post_mat, 0);
    
    for(int i = 0; i < n_pg[g]; i++){
      c_post[i] = c_mean[i]; 
    }
    
    // Phi
    List phi_post = group_term_post_mean(phi_save,nburn);

    // Csi 
    List csi_post = group_term_post_mean(csi_save,nburn); 

    // x 
    List x_post = group_term_post_mean(x_save,nburn); 

    // a0 
    double a0_post = arma::mean(a0_save.subvec(nburn, nrun - 1));

    // c0 
    double c0_post = arma::mean(c0_save.subvec(nburn, nrun - 1));
    
    // Lambda 
    double lambda_post = arma::mean(lambda_save.subvec(nburn, nrun - 1));
    
    // Sigma a 
    double sigma_a_post = arma::mean(sigma_a_save.subvec(nburn, nrun - 1));

    // Sigma c 
    double sigma_c_post = arma::mean(sigma_c_save.subvec(nburn, nrun - 1));
    
    // Sigma eps
    double sigma_eps_post = arma::mean(sigma_eps_save.subvec(nburn, nrun - 1));

    // sd
    List sd_post = individual_term_post_mean_sd_single_group(sd_save,n_pg[g],nburn);
    
    // h
    List h_post = build_h_post_single_group(Bm_phi[g],phi_post,n_pg[g]);
    arma::mat h_mat = build_h_post_cpp_single_group(Bm_phi[g],phi_post,n_pg[g],y_info["n_obs_max"]);
    
    // Bm
    std::vector<arma::mat> Bm_beta_post = build_Bm_single_group(h_mat,n_obs[g],
                            knots_beta,degree_beta,intercept_beta,boundary_beta);

    // y 
    Rcpp::List y_post = build_y_pred_telesca(Bm_beta_post,beta_post,a_post,c_post,n_pg[g]);
    arma::mat y_post_mat = pad_list_of_lists_single_group(y_post,n_obs[g],y_info["n_obs_max"]);
    List y_star = build_y_star_single_group(h_post,y[g],n_obs[g],n_pg[g]); 
    List y_star_smoot = build_y_star_single_group(h_post,y_post_mat,n_obs[g],n_pg[g]);
    
    // Final List 
    List Result_g; 
    
    // Post
    List post; 
    
    post["beta"]        = beta_post; 
    post["a"]           = a_post;
    post["c"]           = c_post;
    post["phi"]         = phi_post;
    post["csi"]         = csi_post; 
    post["x"]           = x_post;
    post["a0"]          = a0_post;
    post["c0"]          = c0_post;
    post["lambda"]      = lambda_post; 
    post["sigma_a"]     = sigma_a_post;
    post["sigma_c"]     = sigma_c_post;
    post["sigma_eps"]   = sigma_eps_post; 
    post["sd"]          = sd_post; 
    post["zeta"]        = zeta; 
    
    Result_g["post"] = post; 
    
    // Full 
    List full; 
    
    Rcpp::List beta_full(1);
    beta_full[0] = beta_save;
    
    Rcpp::List y_original(1);
    y_original[0] = y[g];
  
    Rcpp::List phi_full(n_pg[g]);
    Rcpp::List csi_full(n_pg[g]);
    Rcpp::List x_full(n_pg[g]);
    Rcpp::List a_full(n_pg[g]);
    Rcpp::List c_full(n_pg[g]);
    Rcpp::List sd_full(n_pg[g]);
  
    for (size_t i = 0; i < phi_save.size(); ++i) {
        phi_full[i]    = phi_save[i];
        csi_full[i]    = csi_save[i];
        x_full[i]      = x_save[i];
        a_full[i]      = a_save.col(i);
        c_full[i]      = c_save.col(i);
    }
    
    full["beta"]        = beta_full[0]; 
    full["a"]           = a_full;
    full["c"]           = c_full;
    full["phi"]         = phi_full;
    full["csi"]         = csi_full;
    full["x"]           = x_full;
    full["a0"]          = a0_save;
    full["c0"]          = c0_save;
    full["lambda"]      = lambda_save; 
    full["sigma_a"]     = sigma_a_save;
    full["sigma_c"]     = sigma_c_save;
    full["sigma_eps"]   = sigma_eps_save; 
    full["sd"]          = sd_save; 
    
    Result_g["full"] = full; 
    
    // Acceptance rates
    Result_g["acceptance"] = accept_norm; 
    
    // Warping functions 
    Result_g["h"] = h_post;
    
    // Bm 
    List Bm; 
    
    Bm["Bm_beta"]  = Bm_beta_post;
    Bm["Bm_phi"]   = Bm_phi[g];
    
    Result_g["Bm"] = Bm; 
    
    // y_info 
    List y_info_group_g; 

    y_info_group_g["n_groups"] = 1;
    y_info_group_g["n_patients"] = n_pg[g];
    y_info_group_g["n_obs"] = n_obs[g];
    y_info_group_g["n_obs_max"] = n_obs_tot_per_group[g];
    y_info_group_g["y_val"] = y_val[g];

    Result_g["y_info"] = y_info_group_g;
    
    // y 
    List y_pred; 
    
    y_pred["y"]            = y_original; 
    y_pred["y_smoot"]      = y_post; 
    y_pred["y_star"]       = y_star; 
    y_pred["y_star_smoot"] = y_star_smoot;
    
    Result_g["curves"] = y_pred; 
    
    if(G > 1){

      Rcout << "DONE group " <<g+1 << std::endl;
      Rcout << std::endl;
      Result["group_" + std::to_string(g + 1)] = Result_g;

    } else {
      Rcout << "DONE" << std::endl;
      return Result_g; 
    }

  }
  
  Rcout << "DONE" << std::endl;

  return Result;
}

// [[Rcpp::export]]
List fda_smooth_and_warp_group(Rcpp::List params)
{
    // ==========================
    // Estrazione parametri dalla lista unica
    // ==========================

    if (!params.containsElementNamed("y") || Rf_isNull(params["y"])) {
        Rcpp::stop("'y' must be provided in params list");
    }
    std::vector<arma::mat> y = Rcpp::as<std::vector<arma::mat>>(params["y"]);

    Nullable<int> n_groups              = get_nullable<int>(params, "n_groups");
    Nullable<IntegerVector> n_per_group = get_nullable<IntegerVector>(params, "n_per_group");

    int nburn         = get_or_default<int>(params, "nburn", 100);
    int niter         = get_or_default<int>(params, "niter", 500);
    int n_knots_beta  = get_or_default<int>(params, "n_knots_beta", 10);
    int n_knots_gamma = get_or_default<int>(params, "n_knots_gamma", 3);
    int n_knots_phi   = get_or_default<int>(params, "n_knots_phi", 6);

    Nullable<arma::vec> n_knots_phi_vec = get_nullable<arma::vec>(params, "n_knots_phi_vec");

    Nullable<arma::vec> beta0  = get_nullable<arma::vec>(params, "beta0");
    Nullable<arma::vec> gamma0 = get_nullable<arma::vec>(params, "gamma0");

    double a_eps    = get_or_default<double>(params, "a_eps", 3.0);
    double b_eps    = get_or_default<double>(params, "b_eps", 0.1);
    double a_lambda = get_or_default<double>(params, "a_lambda", 3.0);
    double b_lambda = get_or_default<double>(params, "b_lambda", 0.1);
    double a_gamma  = get_or_default<double>(params, "a_gamma", 3.0);
    double b_gamma  = get_or_default<double>(params, "b_gamma", 0.1);

    double m_c0     = get_or_default<double>(params, "m_c0", 0.0);
    double sigma_c0 = get_or_default<double>(params, "sigma_c0", 10.0);
    double m_a0     = get_or_default<double>(params, "m_a0", 5.0);
    double sigma_a0 = get_or_default<double>(params, "sigma_a0", 10.0);

    double a_c = get_or_default<double>(params, "a_c", 250.0);
    double b_c = get_or_default<double>(params, "b_c", 600.0);
    double a_a = get_or_default<double>(params, "a_a", 25.0);
    double b_a = get_or_default<double>(params, "b_a", 100.0);

    double a_phi          = get_or_default<double>(params, "a_phi", 4.0);
    double b_phi          = get_or_default<double>(params, "b_phi", 4.0);
    double coeff_var_phi  = get_or_default<double>(params, "coeff_var_phi", 0.02);
    Nullable<arma::vec> phi0 = get_nullable<arma::vec>(params, "phi0");

    Nullable<arma::vec> a_f = get_nullable<arma::vec>(params, "a_f");
    double b_f = get_or_default<double>(params, "b_f", 1.0);

    int degree_beta  = get_or_default<int>(params, "degree_beta", 3);
    int degree_gamma = get_or_default<int>(params, "degree_gamma", 3);
    int degree_phi   = get_or_default<int>(params, "degree_phi", 3);

    Nullable<NumericVector> boundary_beta  = get_nullable<NumericVector>(params, "boundary_beta");
    Nullable<NumericVector> boundary_gamma = get_nullable<NumericVector>(params, "boundary_gamma");
    Nullable<NumericVector> boundary_phi   = get_nullable<NumericVector>(params, "boundary_phi");

    bool intercept_beta  = get_or_default<bool>(params, "intercept_beta", false);
    bool intercept_gamma = get_or_default<bool>(params, "intercept_gamma", false);
    bool intercept_phi   = get_or_default<bool>(params, "intercept_phi", false);

    std::string WARP  = get_or_default<std::string>(params, "WARP", std::string("gamma-adaptation"));
    std::string SMOOT = get_or_default<std::string>(params, "SMOOT", std::string("gardella"));

    Nullable<arma::mat> zeta_0 = get_nullable<arma::mat>(params, "zeta_0");
    double zeta_0_coeff = get_or_default<double>(params, "zeta_0_coeff", 0.001);

    Nullable<double> sd0         = get_nullable<double>(params, "sd0");
    Nullable<arma::vec> sd0_land = get_nullable<arma::vec>(params, "sd0_land");
    double sd_min       = get_or_default<double>(params, "sd_min", 0.01);
    double sd_max       = get_or_default<double>(params, "sd_max", 1.0);
    double target_alpha = get_or_default<double>(params, "target_alpha", 0.234);
    double LAMB         = get_or_default<double>(params, "LAMB", 0.6);
    double EPS          = get_or_default<double>(params, "EPS", 1e-6);

    int n0_zeta       = get_or_default<int>(params, "n0_zeta", 10);
    bool INIT_WARP    = get_or_default<bool>(params, "INIT_WARP", false);
    Nullable<arma::vec> phi_init = get_nullable<arma::vec>(params, "phi_init");
    double SOMMA_csi  = get_or_default<double>(params, "SOMMA_csi", 200);
     
    if(WARP == "gamma-adaptation") {

        if (SMOOT == "gardella"){

          Rcout << "Running Gamma-adaptive warping with Gardella smoothing..." << std::endl;
          Rcout << std::endl;

          List result = gardella_gamma_adaptation(
            y = y,n_groups = n_groups,n_per_group = n_per_group,nburn = nburn,niter = niter,n_knots_beta = n_knots_beta,
            n_knots_gamma = n_knots_gamma,n_knots_phi = n_knots_phi,
            beta0 = beta0, gamma0 = gamma0, a_eps = a_eps,b_eps = b_eps,a_lambda = a_lambda,
            b_lambda = b_lambda,a_gamma = a_gamma,b_gamma = b_gamma,a_f = a_f,b_f = b_f,
            degree_beta = degree_beta, degree_gamma = degree_gamma, degree_phi = degree_phi, 
            intercept_beta = intercept_beta, intercept_gamma = intercept_gamma, intercept_phi = intercept_phi,
            boundary_beta = boundary_beta, boundary_gamma = boundary_gamma, boundary_phi = boundary_phi,
            zeta_0 = zeta_0, zeta_0_coeff = zeta_0_coeff,
            sd0 = sd0,sd_min = sd_min,sd_max = sd_max,target_alpha = target_alpha,LAMB = LAMB,EPS = EPS,
            n0_zeta = n0_zeta,INIT_WARP = INIT_WARP,phi_init = phi_init,SOMMA_csi = SOMMA_csi);

          return result;
        }
        else if(SMOOT == "telesca"){

          Rcout << "Running Gamma-adaptive warping with Telesca smoothing..." << std::endl;
          Rcout << std::endl;

          List result = telesca_gamma_adaptation(
          y = y,n_groups = n_groups,n_per_group = n_per_group,nburn = nburn,niter = niter,
          n_knots_beta = n_knots_beta,n_knots_phi = n_knots_phi,
          beta0 = beta0, m_c0 = m_c0, sigma_c0 = sigma_c0, m_a0 = m_a0, sigma_a0 = sigma_a0,
          a_c = a_c, b_c = b_c, a_a = a_a, b_a = b_a,      
          a_eps = a_eps,b_eps = b_eps,a_lambda = a_lambda,b_lambda = b_lambda, a_f = a_f,b_f = b_f,
          zeta_0 = zeta_0, zeta_0_coeff = zeta_0_coeff,
          sd0 = sd0,sd_min = sd_min,sd_max = sd_max,target_alpha = target_alpha,LAMB = LAMB,EPS = EPS,
          n0_zeta = n0_zeta,INIT_WARP = INIT_WARP,phi_init = phi_init,SOMMA_csi = SOMMA_csi,
          degree_beta = degree_beta, degree_phi = degree_phi, 
          intercept_beta = intercept_beta, intercept_phi = intercept_phi,
          boundary_beta = boundary_beta, boundary_phi = boundary_phi);
          
          return result;


        }
        else {Rcpp::stop("Smoothing type not valid: " + std::string(SMOOT));}
      } 
  
    else if (WARP == "gaussian") {

        if(SMOOT == "gardella"){
          Rcout << "Running Gaussian warping with Gardella smoothing..." << std::endl;
          Rcout << std::endl;

          List result = gardella_gaussian(
          y = y,n_groups = n_groups,n_per_group = n_per_group,nburn = nburn,niter = niter,n_knots_beta = n_knots_beta,
          n_knots_gamma = n_knots_gamma,n_knots_phi = n_knots_phi,
          beta0 = beta0, gamma0 = gamma0, a_eps = a_eps,b_eps = b_eps,a_lambda = a_lambda,
          b_lambda = b_lambda,a_gamma = a_gamma,b_gamma = b_gamma,a_phi = a_phi,b_phi = b_phi,coeff_var_phi = coeff_var_phi,
          phi0 = phi0, INIT_WARP = INIT_WARP,phi_init = phi_init,
          degree_beta = degree_beta, degree_gamma = degree_gamma, degree_phi = degree_phi, 
          intercept_beta = intercept_beta, intercept_gamma = intercept_gamma, intercept_phi = intercept_phi,
          boundary_beta = boundary_beta, boundary_gamma = boundary_gamma, boundary_phi = boundary_phi);

          return result;
        } else if (SMOOT == "telesca"){

          Rcout << "Running Gaussian warping with Telesca smoothing..." << std::endl;
          Rcout << std::endl;

          List result = telesca_gaussian(
          y = y,n_groups = n_groups,n_per_group = n_per_group,nburn = nburn,niter = niter,
          n_knots_beta = n_knots_beta,n_knots_phi = n_knots_phi,
          beta0 = beta0, m_c0 = m_c0, sigma_c0 = sigma_c0, m_a0 = m_a0, sigma_a0 = sigma_a0,
          a_c = a_c, b_c = b_c, a_a = a_a, b_a = b_a,      
          a_eps = a_eps,b_eps = b_eps,a_lambda = a_lambda,
          b_lambda = b_lambda,a_phi = a_phi,b_phi = b_phi,coeff_var_phi = coeff_var_phi,
          phi0 = phi0, INIT_WARP = INIT_WARP,phi_init = phi_init,
          degree_beta = degree_beta, degree_phi = degree_phi, 
          intercept_beta = intercept_beta, intercept_phi = intercept_phi,
          boundary_beta = boundary_beta, boundary_phi = boundary_phi);
          
          return result;
        } else { Rcpp::stop("Smoothing type not valid: " + std::string(SMOOT));}
      }
    else {
      {Rcpp::stop("Warping type not valid: " + std::string(SMOOT));}
  }
}


