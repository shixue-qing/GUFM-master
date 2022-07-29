clear all; close all; clc
%part_dir = './chair/';
%part_name = 'chair_0988';       %改
a=dir('F:\数据集\shrec16\尝试off_mat乘100\2\*.mat');
for i=1:length(a)

A = load(fullfile(( 'F:\数据集\shrec16\尝试off_mat乘100\2\'),a(i).name));
%A = load('./chair/chair_0988'); %改
%[r c]=size(A.VERT);
%TRIV = A.TRIV;
%m = A.m;
%B=A.VERT(1:1916,:);   %改
%VERT = B;
%[n k]=size(VERT);
model_evecs = A.xyz_kuosan_hks1(:,4:153);
%save(['F:\数据集\shrec16\尝试off_mat乘100\chair\',part_name,'.mat'],'TRIV','VERT','m','n');
save(['F:\数据集\shrec16\尝试off_mat乘100\2\',a(i).name],'model_evecs') ;
end