%%% This code to implement IF-PCA is from the paper: Jin, J. and Wang, W. (2016). Influential Features PCA for High Dimensional Clustering, to appear with discussion. 

function [err, stats, numselect, data_select, V] = ifpca_paper(Data, Class, K, KSvalue, pvalcut, kmeansrep, Nrep, per) 
%Output: 
%
%s: number of features selected in IF-PCA
%err: error rate for IFPCA(err.if), kmeans(err.kmeans), 
%kmeans++(err.kmeanspp, change the function below for kmeanspp!!!!!), Hiearchical clustering with 'complete' package
%(err.com)
%stats: statistics for each feature, including KS scores(stats.KS),
%P-values(stats.pval), HC functionals(stats.HC), feature ranks(stats.ranking)
%numselect: number of selected features in IF-PCA
%data_select: p'-by-n selected IF data matrix
%V: n-by-(K-1) IF-PCA(no Kmeans) data matrix

[p, n] = size(Data);

if (nargin<4||isempty(KSvalue))
    nullsimu = true; 
else 
    nullsimu = false;
end

if (nargin<5)||isempty(pvalcut)
    pvalcut = log(p)/p;
end

if (nargin<6)||isempty(kmeansrep)
    %kmeansrep = 30;
    kmeansrep = 1;
end

if (nargin<7)||isempty(Nrep)
    Nrep = 20;
end

if (nargin<8)||isempty(per)
    per = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Normalize Data
gm = mean(Data'); gsd = std(Data');
Data = (Data - repmat(gm', 1, n))./repmat(gsd', 1, n);

%Simulate KS values
if (nullsimu)
    rep = 100*p;
    KSvalue = zeros(rep,1); 
    parfor i = 1:rep
        x = randn(n,1); 
	    z = (x - mean(x))/std(x);
	    z = z/sqrt(1 - 1/n);
	    pi = normcdf(z);
	    pi = sort(pi);
        kk = (0:n)'/n;
	    KSvalue(i) = max(max(abs(kk(1:n) - pi)), max(abs(kk(2:(n+1)) - pi)));
    end
    KSvalue = KSvalue*sqrt(n);
end
KSmean = mean(KSvalue); KSstd = std(KSvalue);
if(per < 1)
    KSvaluesort = sort(KSvalue, 'ascend');
    KSmean = mean(KSvaluesort(1:round(per*rep)));
    KSstd = std(KSvaluesort(1:round(per*rep)));
end
%Calculate KS value for each feature in the data set
KS = zeros(p, 1);
parfor j= 1:p
    pi = normcdf(Data(j,:)/sqrt(1 - 1/n));
    pi = sort(pi);
    kk = (0:n)'/n;
    KS(j) = max(max(abs(kk(1:n) - pi')), max(abs(kk(2:(n+1)) - pi')));
end
if(per == 1)
    KS = (KS - mean(KS))/std(KS); 
else
    KSsort = sort(KS, 'ascend');
    KS = (KS - mean(KSsort(1:round(per*p))))/std(KSsort(1:round(per*p)));
end
%Calculate P-value with simulated KS values
KSadjust = KS*KSstd + KSmean;
pval = zeros(p,1);
parfor i = 1:p
    pval(i) = mean(KSvalue > KSadjust(i));
end
[psort, ranking] = sort(pval, 'ascend');%ranking: index in the unsorted array in the sorted one, B = sort(A), B = A(ranking)

% Calculate HC functional at each data point
kk = (1:p)'/(1 + p);
HCsort = sqrt(p)*(kk - psort)./sqrt(kk);
HCsort  =   HCsort./sqrt(max(sqrt(n)*(kk - psort)./kk, 0) + 1);
HC = zeros(p,1);
HC(ranking) = HCsort;

% Decide the threshold
Ind = find(psort>pvalcut, 1, 'first');
ratio = HCsort;
ratio(1:Ind-1) = -Inf; ratio(round(p/2)+1:end)=-Inf;
L = find(ratio == max(ratio), 1, 'last');
numselect = L; 

% Record the statistics for every feature
stats.KS = KS; stats.HC = HC; stats.pval = pval; stats.ranking = ranking; 

% IF-PCA
data_select = Data(pval <= psort(L), :); % m-by-n selected IF data
G = data_select'*data_select;
[V, ~] = eigs(G, K - 1); 
errors = zeros(Nrep, 1);
parfor jj=1:Nrep
    class1a = kmeans(V, K, 'replicates', kmeansrep);
    t = crosstab(class1a, Class);
    if K == 2
        errors(jj) = min(sum(diag(t)), n - sum(diag(t)))/n;
    else
        errors(jj) = errors_calculate(t);
    end
end
err.IFPCA = errors; 

%{
% Classical PCA
G = Data'*Data;
errors = zeros(Nrep, 1);
parfor jj=1:Nrep
    [V, ~] = eigs(G, K - 1); class1a = kmeans(V, K, 'replicates', kmeansrep);
    t = crosstab(class1a, Class);
    if K == 2
        errors(jj) = min(sum(diag(t)), n - sum(diag(t)))/n;
    else
        errors(jj) = errors_calculate(t);
    end
end
err.PCA = errors; 

% K-means 
errors = zeros(Nrep, 1); 
parfor jj = 1:Nrep
    class1a = kmeans(Data', K, 'replicates', kmeansrep); %kmeans should be kmeanspp
    t = crosstab(class1a, Class);
    if K == 2
        errors(jj) = min(sum(diag(t)), n - sum(diag(t)))/n;
    else
        errors(jj) = errors_calculate(t);
    end
end
err.kmeans = errors;

% K-means++  
errors = zeros(Nrep, 1); 
err.kmeansplus = errors;
%}


