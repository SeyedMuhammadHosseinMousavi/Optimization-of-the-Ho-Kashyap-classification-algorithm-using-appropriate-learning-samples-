clear all ; close all ; clc ;

filename=strcat('E:\My File\UNI\My\DataSet\ionosphere.data.txt');
DataSet=importdata(filename);

save('DataSet','DataSet');