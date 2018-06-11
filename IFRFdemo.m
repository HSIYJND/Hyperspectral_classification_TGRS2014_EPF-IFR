clc,clear

addpath('./libsvm/');
addpath('./drtoolbox/');
addpath('./drtoolbox/techniques');

no_classes       = 16;
no_train         = round(1024);
%%%%%
% load the ground truth and the hyperspectral image
path='.\Dataset\';
inputs = 'IndiaP';
location = [path,inputs];
load (location);
%%% size of image 
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';
%%% construct traing and test dataset
indexes=train_test_random_new(GroundT(2,:),fix(no_train/no_classes),no_train);
%%% image fusion
img2=average_fusion(img,20);
%%% normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
%%% IFRF feature construction
fimg=spatial_feature(fimg,200,0.3);

%%% SVM classification
fimg = ToVector(fimg);
fimg = fimg';
fimg=double(fimg);
%%%
train_SL = GroundT(:,indexes);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';
% Normalizing Training and original img 
[train_samples,M,m] = scale_func(train_samples);
[fimg ] = scale_func(fimg',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg,model); 
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA,AA,kappa,CA]=confusion(GroudTest,ResultTest)
Result = reshape(Result,no_lines,no_rows);
VClassMap=label2color(Result,'india');
figure,imshow(VClassMap);
