
%%% Load microarray data sets
% brain
% breast
% colon
% leukemia
% lung1
% lung2
% lymphoma
% prostate
% srbct
% su
name = ‘brain’;
Data = load(strcat(brain, '.x.txt')); %p-by-n
Class = load(strcat(name, '.y.txt'));

%%% Load single-cell RNA-seq data sets
% camp1
% camp2
% darmanis
% deng
% goolam
% grun
% li
% patel
name = ‘camp1’;
DataRaw = load(strcat(name, ‘-x-filter.txt')); %p-by-n
Class = load(strcat(name, '-y.txt'));



%%% Run IF-PCA
[p, n] = size(Data);
K = max(Class) - min(Class) + 1;
[err, stats, numselect, data_select, V] = ifpca_paper(Data, Class, K); 



%%% Measure clustering errors
err.IFPCA * n %clustering errors
1 - mean(err.IFPCA) %clustering accuracy

