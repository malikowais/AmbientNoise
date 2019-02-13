
clc
clear
isb =  xlsread('ISB_Project.xlsx','Data');

% X = isb(:,3:6);
X = isb(:,3:5);
X(147, :) = [];
rng default

clust = zeros(size(X,1),6);
for i=1:6
clust(:,i) = kmeans(X,i,'emptyaction','singleton','replicate',10,'Display','final');
end

eva1 = evalclusters(X,clust,'CalinskiHarabasz')
eva2 = evalclusters(X,clust,'DaviesBouldin')
eva3 = evalclusters(X,clust,'silhouette')

plot(eva1)
plot(eva2)
plot(eva3)

silhouette(X,clust(:,3))

scatter3(X(:,1),X(:,2),X(:,3),50,clust(:,3),'filled')

all_clusters = xlsread('ISB_Project.xlsx', 'K_Means');
% ind1 = ward_eu(:,9) == 1;
% ind2 = ward_eu(:,9) == 2;
% cluster1 = ward_eu(ind1,:);
% cluster2 = ward_eu(ind2,:);

%%% For kmeans 3 clusters
[p1,tbl1,stats1] = kruskalwallis(all_clusters(:,1),all_clusters(:,4))
[p2,tbl2,stats2] = kruskalwallis(all_clusters(:,2),all_clusters(:,4))
[p3,tbl3,stats3] = kruskalwallis(all_clusters(:,3),all_clusters(:,4))

c1 = multcompare(stats1)
c2 = multcompare(stats2)
c3 = multcompare(stats3)
