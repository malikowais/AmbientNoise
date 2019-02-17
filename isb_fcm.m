
clc
clear
isb =  xlsread('ISB_Project.xlsx','Data');

% X = isb(:,3:6);
X = isb(:,3:5);
%X(147, :) = [];
rng default

clust = zeros(size(X,1),6);

clust(:,1) = 1;

[centers2,U2] = fcm(X,2);
maxU2 = max(U2);
index12 = find(U2(1,:) == maxU2);
index22 = find(U2(2,:) == maxU2);

clust(index12,2) = 1;
clust(index22,2) = 2;

[centers3,U3] = fcm(X,3);
maxU3 = max(U3);
index13 = find(U3(1,:) == maxU3);
index23 = find(U3(2,:) == maxU3);
index33 = find(U3(3,:) == maxU3);

clust(index13,3) = 1;
clust(index23,3) = 2;
clust(index33,3) = 3;

[centers4,U4] = fcm(X,4);
maxU4 = max(U4);
index14 = find(U4(1,:) == maxU4);
index24 = find(U4(2,:) == maxU4);
index34 = find(U4(3,:) == maxU4);
index44 = find(U4(4,:) == maxU4);

clust(index14,4) = 1;
clust(index24,4) = 2;
clust(index34,4) = 3;
clust(index44,4) = 4;

[centers5,U5] = fcm(X,5);
maxU5 = max(U5);
index15 = find(U5(1,:) == maxU5);
index25 = find(U5(2,:) == maxU5);
index35 = find(U5(3,:) == maxU5);
index45 = find(U5(4,:) == maxU5);
index55 = find(U5(5,:) == maxU5);

clust(index15,5) = 1;
clust(index25,5) = 2;
clust(index35,5) = 3;
clust(index45,5) = 4;
clust(index55,5) = 5;


[centers6,U6] = fcm(X,6);
maxU6 = max(U6);
index16 = find(U6(1,:) == maxU6);
index26 = find(U6(2,:) == maxU6);
index36 = find(U6(3,:) == maxU6);
index46 = find(U6(4,:) == maxU6);
index56 = find(U6(5,:) == maxU6);
index66 = find(U6(6,:) == maxU6);

clust(index16,6) = 1;
clust(index26,6) = 2;
clust(index36,6) = 3;
clust(index46,6) = 4;
clust(index56,6) = 5;
clust(index66,6) = 6;

eva1 = evalclusters(X,clust,'CalinskiHarabasz')
eva2 = evalclusters(X,clust,'DaviesBouldin')
eva3 = evalclusters(X,clust,'silhouette')

plot(eva1)
plot(eva2)
plot(eva3)

silhouette(X,clust(:,3))

scatter3(X(:,1),X(:,2),X(:,3),50,clust(:,3),'filled')
