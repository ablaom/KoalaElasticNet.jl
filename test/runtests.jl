using Koala
using KoalaElasticNet
using Base.Test

# Load some data and define train/test rows:
const X, y = load_ames();
y = log.(y)

const train, test = split(eachindex(y), 0.7); # 70:30 split

# Instantiate a model:
elastic = ElasticNetRegressor(lambda_min_ratio=1e-9)
showall(elastic)

# Change the default transformer behaviour to include Box-Cox
# transformations on input features:
tX = default_transformer_X(elastic)
showall(tX)
tX.boxcox = true

# Build a machine that uses the modified transformer:
elasticM = Machine(elastic, X, y, train, transformer_X=tX, verbosity=3)

# Train the model:
fit!(elasticM, train)
showall(elasticM)

rpt = elasticM.report

m = rpt[:loglambdaopt]
delta = rpt[:loglambdaopt_stde]
lambdas = exp.(linspace(m - delta, m + delta, 10))

# choose range for l1/l2 ratio:
alphas = linspace(0.8,1.0,15)

# tune using cross-validation
alphas, lambdas, rmserrors = @curve α alphas λ lambdas begin
    elastic.alpha, elastic.lambda = α, λ
    mean(cv(elasticM, train, parallel=true, n_folds=9, verbosity=0))
end

# set regularization parameters to optimal values:
j, k = ind2sub(size(rmserrors), indmin(rmserrors))
elastic.alpha, elastic.lambda = alphas[j], lambdas[k]

# retrain:
fit!(elasticM)

# report score:
score = err(elasticM, test)
@test score < 0.14 && score >0.12 
