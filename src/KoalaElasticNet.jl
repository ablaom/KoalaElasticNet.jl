module KoalaElasticNet

# new:
export ElasticNetRegressor

# needed in this module:
import Koala: Regressor, BaseType, keys_ordered_by_values, SupervisedMachine, params
import Koala: Transformer
import KoalaTransforms
import Lasso
import DataFrames: AbstractDataFrame, DataFrame
import UnicodePlots

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, transform, inverse_transform

# development only:
# using ADBUtilities

## Helpers

"""
`getcol(A, j)` gets the jth column of a sparse matrix `A`, returned as
a sparse vector. If `j=0` then the last column is returned. If `j`
exceeds the number of columns of `A`, then a zero vector is returned.

"""
function getcol(A::SparseMatrixCSC{Float64,Int}, j::Int)
    if j == 0
        j = A.n
    end
    j > A.n ? (return SparseVector(A.m,Int[],Float64[])) : nothing
    I = Int[]     # indices for non-zero entries of col
    V = Float64[] # values of non-zero entries of col
    for k in A.colptr[j]:(A.colptr[j+1] - 1)
        push!(I, A.rowval[k])
        push!(V, A.nzval[k])
    end
    return SparseVector(A.m,I,V)
end

# function to determine the largest lambda value conceivable for given
# alpha and training data:
function lambda_max(X, y, alpha)
    Xy = X'*(y .- mean(y)) # dot products of feature columns with centred response
    λmax = abs(Xy[1])
    for i = 2:length(Xy)
        x = abs(Xy[i])
        if x > λmax
            λmax = x
        end
    end
    return λmax/(alpha*length(y))
end


## Model type definitions

struct LinearPredictor <: BaseType
    intercept::Float64
    coefs::SparseVector{Float64,Int64}
end

# Following returns a `DataFrame` with three columns:
#
# column name | description
# :-----------|:-------------------------------------------------
# `:index`    | index of a feature used to train `predictor`
# `:feature`  | corresponding feature label provided by `features`
# `:coef`     | coefficient for that feature in the predictor
#
# The rows are ordered by the absolute value of the coefficients. If
# `rgs` is unfitted, an error is returned.
function coef_info(predictor::LinearPredictor, features)
    coef_given_index = Dict{Int, Float64}()
    abs_coef_given_index = Dict{Int, Float64}()
    v = predictor.coefs # SparseVector
    for k in eachindex(v.nzval)
        coef_given_index[v.nzind[k]] = v.nzval[k]
        abs_coef_given_index[v.nzind[k]] = abs(v.nzval[k])
    end
    df = DataFrame()
    df[:index] = reverse(keys_ordered_by_values(abs_coef_given_index))
    df[:feature] = map(df[:index]) do index
        features[index]
    end
    df[:coef] = map(df[:index]) do index
        coef_given_index[index]
    end
    return df
end

mutable struct ElasticNetRegressor <: Regressor{LinearPredictor}

    lambda::Float64
    n_lambdas::Int
    alpha::Float64
    max_n_coefs::Int
    criterion::Symbol
    lambda_min_ratio::Float64

    # for cv-optimizing lambda:
    n_folds::Int

    function ElasticNetRegressor(lambda, n_lambdas::Int, alpha, 
                                 max_n_coefs::Int, criterion::Symbol,
                                 lambda_min_ratio, n_folds::Int)
        if alpha <= 0.0 || alpha > 1.0
            if alpha == 0.0
                throw(Base.error("alpha=0 dissallowed."*
                                 " Consider ridge regression instead."))
            end
            throw(DomainError)
        end
        return new(lambda, n_lambdas, alpha, max_n_coefs, criterion,
                  lambda_min_ratio, n_folds)
    end

end

# lazy keywork constructor
ElasticNetRegressor(;lambda=0.0, n_lambdas::Int=100,
                    alpha=1.0,
                    max_n_coefs::Int=0, criterion::Symbol=:coef,
                    lambda_min_ratio=0.0, n_folds::Int=9) = 
                        ElasticNetRegressor(lambda, n_lambdas,
                                            alpha,
                                            max_n_coefs, criterion,
                                            lambda_min_ratio, n_folds)

# `showall` method for `ElasticNetRegressor` machines:
function Base.showall(stream::IO,
                      mach::SupervisedMachine{LinearPredictor, ElasticNetRegressor})
    show(stream, mach)
    println(stream)
    if isdefined(mach,:report) && :feature_importance_curve in keys(mach.report)
        features, importance = mach.report[:feature_importance_curve]
        plt = UnicodePlots.barplot(features, importance,
              title="Feature importance (coefs of linear predictor)")
    end
    dict = params(mach)
    report_items = sort(collect(keys(dict[:report])))
    dict[:report] = "Dict with keys: $report_items"
    dict[:Xt] = string(typeof(mach.Xt), " of shape ", size(mach.Xt))
    dict[:yt] = string(typeof(mach.yt), " of shape ", size(mach.yt))
    delete!(dict, :cache)
    showall(stream, dict)
    println(stream, "\nModel detail:")
    showall(stream, mach.model)
    if isdefined(mach,:report) && :feature_importance_curve in keys(mach.report)
        show(stream, plt)
    end
end

default_transformer_X(model::ElasticNetRegressor) =
    KoalaTransforms.DataFrameToArrayTransformer()
default_transformer_y(model::ElasticNetRegressor) =
    KoalaTransforms.RegressionTargetTransformer()
    
# Note: For readability we now use `X` and `y` in place of `Xt` and `yt`

struct Cache <: BaseType
    X::Matrix{Float64}
    y::Vector{Float64}
    features::Vector{Symbol}
end

setup(model::ElasticNetRegressor, X, y, scheme_X, parallel, verbosity) =
    Cache(X, y, scheme_X.spawned_features)

# The core elastic net algorithm does not fit to model parameters but
# fits a *sequence* (aka path) of predictors corresponding to a
# specified sequence of `lambda` values (all other model parameters
# fixed). The reason is this: for high `lambda` values, the optimal
# predictor is a trivial (constant) linear predictor, and it is
# efficient to compute predictors for slightly lower values as
# perturbations to this model, and so on as `lambda` is slowly
# decreased.

# The following core function returns the optimal predictor (really a
# *path* of predictors) for a specified path of lambda values,
# `lambdas`. If `lambdas` is empty, it uses an automatically generated
# path instead.

function fit_path(model::ElasticNetRegressor, cache, rows, lambdas)

    X = cache.X[rows,:]
    y = cache.y[rows]

    if isempty(lambdas)
        if model.max_n_coefs == 0
            if model.lambda_min_ratio == 0.0
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              nλ=model.n_lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              criterion=model.criterion, verbose=false)
            else
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              nλ=model.n_lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              criterion=model.criterion, verbose=false,
                              λminratio=model.lambda_min_ratio)
            end
        else
            if model.lambda_min_ratio == 0.0
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              nλ=model.n_lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              maxncoef=model.max_n_coefs, criterion=model.criterion,
                              verbose=false)
            else
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              nλ=model.n_lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              maxncoef=model.max_n_coefs, criterion=model.criterion,
                              verbose=false, λminratio=model.lambda_min_ratio)
            end
        end
    else
        reverse!(sort!(lambdas))
        if model.max_n_coefs == 0
            if model.lambda_min_ratio == 0.0
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              λ =lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              criterion=model.criterion, verbose=false)
            else
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              λ =lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              criterion=model.criterion, verbose=false,
                              λminratio=model.lambda_min_ratio)
            end
        else
            if model.lambda_min_ratio == 0.0
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              λ =lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              maxncoef=model.max_n_coefs,
                              criterion=model.criterion, verbose=false)
            else
                predictors =
                    Lasso.fit(Lasso.LassoPath, X, y;
                              λ =lambdas, α=model.alpha,
                              standardize=false, # in Koala any standardization is external to fit
                              maxncoef=model.max_n_coefs,
                              criterion=model.criterion,
                              verbose=false, λminratio=model.lambda_min_ratio)
            end
        end
    end
    
    return predictors::Lasso.LassoPath

end

# for generating predictions on a pattern, for each lambda in the path
# used to generate `predictors`:
predict(predictors::Lasso.LassoPath, pattern) =
    vec(Lasso.predict(predictors, pattern'))

function fit(model::ElasticNetRegressor, cache, add, parallel, verbosity)

    report = Dict{Symbol, Any}()

    rows = eachindex(cache.y)

    if model.lambda == 0

        n_folds = model.n_folds
    
        best_lambdas = Array{Float64}(n_folds)
        errors = Array{Float64}(n_folds)
        n_samples = length(rows)
        
        k = floor(Int, n_samples/n_folds)
        first = 1       # first test_bag index
        second = k
        verbosity < 1 ||
            println("Optimizing regularization parameter using "*
                    "$(n_folds)-fold cross-validation. ")
        for n in 1:n_folds
            test_rows = rows[first:second]
            train_rows = vcat(rows[1:first-1], rows[second+1:end])
#            @dbg length(test_rows) length(train_rows) 
            first = first + k 
            second = second + k
            
            # get predictors along an automatically generated lambda-path:
            predictors = fit_path(model, cache, train_rows, Float64[])
            lambdas = predictors.λ
#            @dbg lambdas

            # get RMS errors on the current test set for each lambda in path:
            ss_deviations = zeros(length(lambdas))
            for i in test_rows
                ss_deviations = ss_deviations +
                    map(predict(predictors, cache.X[i,:])) do yhat
                        (yhat - cache.y[i])^2
                    end
            end

            # get lambda with minimum error:
            L = indmin(ss_deviations)
            if L == length(lambdas) && verbosity > 0
                Base.warn("Optimal lambda not reached on descent.\n"*
                          "Consider lowering lambda lambda_min_ratio.")
            end
            best_lambdas[n] = lambdas[L]
            verbosity < 1 || println("fold: $n  optimal lambda: $(lambdas[L])")
            errors[n] = sqrt(ss_deviations[L]/length(test_rows))
        end
        
        loglambdas = log.(best_lambdas)

        report[:loglambdaopt] = mean(loglambdas)
        report[:loglambdaopt_stde] = std(loglambdas)
        report[:lambdaopt] = exp.(report[:loglambdaopt])
        report[:cv_rmse] = mean(errors)
        report[:cv_rmse_stde] = std(errors)

        verbosity < 1 ||
            println("\nOptimal lambda = $(report[:lambdaopt]). See machine "*
                    "report for details.")
        
        # prepare for final train:
        lambda = report[:lambdaopt]
        
    else
        
        lambda = model.lambda 

    end
    
    # calculate lambda-path `lambdas` for final train
    λmax = lambda_max(cache.X, cache.y, model.alpha)
    λmax >= model.lambda ||
        throw(Base.error("Something wrong here"))
    lambdas = exp.(linspace(log(λmax), log(lambda), model.n_lambdas))
    
    # make sure last element of lambdas is *exactly* lambda:
    lambdas[end]=lambda
    
    # final train on all the data:
    predictors = fit_path(model, cache, rows, lambdas)
    
    # check path finishes at optimal lambda:
    if predictors.λ[end] != lambda && verbosity < 1
        verbosity < 1 || Base.warn("Early stopping of path before optimal "*
                                   "or prescribed lambda reached.")
    end
        
    # assemble predictor:
    intercept =  predictors.b0[end]
    coefs = getcol(predictors.coefs, 0) # 0 gets last column
    predictor = LinearPredictor(intercept, coefs)

    # report on the relative strength of each feature in the predictor:
    cinfo = coef_info(predictor, cache.features) # a DataFrame object
    u = Symbol[]
    v = Float64[]
    for i in 1:size(cinfo, 1)
        feature, coef = (cinfo[i, :feature], cinfo[i, :coef])
        coef = floor(1000*coef)/1000
        if coef < 0
            label = string(feature, " (-)")
        else
            label = string(feature, " (+)")
        end
        push!(u, label)
        push!(v, abs(coef))
    end
    report[:feature_importance_curve] = u, v

    return predictor, report, cache
    
end

function predict(predictor::LinearPredictor, pattern::Vector{Float64})

    ret = predictor.intercept
    for i in eachindex(predictor.coefs.nzval)
        ret = ret + predictor.coefs.nzval[i]*pattern[predictor.coefs.nzind[i]]
    end
    
    return ret
    
end

predict(model::ElasticNetRegressor, predictor::LinearPredictor,
        X::Matrix{Float64}, parallel, verbosity) =
            Float64[predict(predictor, X[i,:]) for i in 1:size(X,1)]

end # of module




