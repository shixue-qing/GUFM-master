clear all; close all; clc

%% 加载路径


a=dir('F:\数据集\模型1\2011分类\第三类zuo_linyums\*.mat');
for fi=1:length(a)
% fi = 1;    % 你需要从1开始到数据集总个数    改
load(fullfile(( 'F:\数据集\模型1\2011分类\第三类zuo_linyums\'),a(fi).name));             %改

% D = pdist2(VERT,VERT,'mahalanobis');cosine
O_juli = pdist2(D,D,'mahalanobis');

hmin=min(min(O_juli));
hmax=max(max(O_juli));
DD =(O_juli-hmin)*(1-0)/(hmax-hmin);
% D = pdist2(D,D);

save(['F:\数据集\模型1\2011分类\第三类zuo_linyums_ms\',a(fi).name],'DD');
disp(a(fi).name)
end


% clear all; close all; clc
% 
% %% 加载路径
% 
% 
% a=dir('E:\各论文实现代码\uns_dfm\uns_master\uns_master\Single Pair Experiment\artist_models\*.mat');
% fi = 2;
% % fi = 1;    % 你需要从1开始到数据集总个数    改
% load(fullfile(( 'E:\各论文实现代码\uns_dfm\uns_master\uns_master\Single Pair Experiment\artist_models\'),a(fi).name));             %改
% 
% D = pdist2(model.VERT,model.VERT,'mahalanobis');
% % D = pdist2(D,D);
% 
% save(['F:\数据集\模型1\2011分类\ysj_ms1\',a(fi).name],'D');
% disp(a(fi).name)
