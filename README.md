# KoalaElasticNet


From the documentation of my old api. Needs adapting to Koala


#### `ElasticNetRegressor(lambda=0, alpha=1.0, standardize=false, max_n_coefs=0, criterion=:ceof)`

ScikitLearn style Wrapper for elastic net implementation at:

[1] http://lassojl.readthedocs.io/en/latest/lasso.html

Algorithm details: 

[2] Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization
paths for generalized linear models via coordinate descent. Journal of
Statistical Software, 33(1), 1.

## Example use

For input patterns, presented as a `DataFrame` or `DataTable` object
`X`, and corresponding output responses `y`:

    julia> rgs = ElasticNetRegressor(alpha=0.5)
    julia> fit!(rgs,X,y) # fit and optimize regularization parameter
    julia> cv_error(rgs, X,y)
    0.534345

    julia> rgs.lambdaopt
    0.005987423  # estimate of optimal value of lambda

    julia> rgs2 = ElasticNetRegressor(X, y, lambda = 0.006, alpha =0.2) # one line construct and train

## Feature extraction

From a trained object `rgs` one may obtain a ranking of the features used in training based on the absolute values of coefficients in the linear model obtained (generally choosing `alpha=1.0` for pure L1 regularization):

    julia> coefs(rgs)
    15×3 DataFrames.DataFrame
    │ Row │ feature │ name                   │ coef       │
    ├─────┼─────────┼────────────────────────┼────────────┤
    │ 1   │ 7       │ SaleType__CWD          │ 0.584603   │
    │ 2   │ 16      │ SaleCondition__Alloca  │ -0.307277  │
    │ 3   │ 11      │ SaleType__New          │ 0.274558   │
    │ 4   │ 19      │ SaleCondition__Partial │ 0.245157   │
    │ 5   │ 15      │ SaleCondition__AdjLand │ -0.221836  │
    │ 6   │ 18      │ SaleCondition__Normal  │ 0.107383   │

Or type `showall(rgs)` in the REPR for a graphical representation. 

## Hyperparameters:

- `lambda`: regularization parameter, denoted λ in [2]. If set to 0
  (default) then, upon fitting, a value is optimized as follows: For
  each training set in a cross-validation regime, the value of
  `lambda` minimizing the RMS error on the hold-out set is noted. The
  minimization is over a "regularization path" of `lambda` values
  (numbering `n_lambda`) which are log-distibuted from λmax down to
  lambda_min_ratio \* λmax. Here λmax is the smallest amount of regularization
  giving a null model.  The geometric mean of these individually
  optimized values is used to train the final model on all available
  data and is stored as the post-fit parameter `lambdaopt`. N.B. *The
  hyperparameter `lambda` is not updated (remains 0).*

- `n_lambdas`: number of `lambda` values tried in each
  cross-validation path if `lambda` is unspecified (initially set to
  0). Defaults to 100.

- `alpha`: denoted α in [2], taking values in (0,1]; measures degree of Lasso in the
  Ridge-Lasso mix and defaults to 1.0 (Lasso regression)

- `standardize`: when `true` historical target values are centred and rescaled to
  have mean 0 and std 1 in the specified training regime. Default: true

- `max_n_coefs`: maximum number of non-zero coefficients being
  admitted. If set to zero, then `max_n_coefs = min(size(X, 2),
  2*size(X, 1))` is used. If exceeded an error is thrown.

- `lambda_min_ratio`: see discussion of `lambda` above. If unspecified
  (or 0.0) then this is automatically selected.

- `criterion`: Early stopping criterion in building paths. If
  `criterion` takes on the value `:coef` then the model is considered
  to have converged if the the maximum absolute squared difference in
  coefficients between successive iterations drops below a certain
  tolerance. This is the criterion used by glmnet. Alternatively, if
  `criterion` takes on the value `:obj` then the model is considered
  to have converged if the the relative change in the Lasso/Elastic
  Net objective between successive iterations drops below a certain
  tolerance. This is the criterion used by GLM.jl. Defaults to `:coef`

## Post-fit parameters:

- `lambdaopt` the optimized value of the regularization parameter, or
  `lambda` if the latter is specified and non-zero.

- `intercept`: intercept of final linear model

- `coefs`:  coefficients for final linear model, stored as `SparseVector{Float64,Int64}`

- `loglambdaopt = log(lambdaopt)`

## If `lambda` is not specified or zero, then these parameters are also available:

- `loglambdaopt_stde`: standard error of cross-validation estimates of
  log(lambda) 

- `cv_rmse`: mean of cross-validation RMS errors 

- `cv_rmse_stde`: standard error of cross-validation RMS errors 


