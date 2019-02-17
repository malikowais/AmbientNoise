clc
clear
isb =  xlsread('ISB_Project.xlsx','Data');

% X = isb(:,3:6);
X = isb(:,3:5);
%X(147, :) = [];
%%% Compute different distance measures

Yeu = pdist(X,'euclidean');
Ych = pdist(X,'chebychev');

%%% Compute linkages for each of the above distance measures

Zaveu = linkage(Yeu,'average');
Zcmeu = linkage(Yeu,'complete');



Zavch = linkage(Ych,'average');

%%% Display Dendrogram for each type of linkage

dendrogram(Zaveu) %% 3 clusters
dendrogram(Zcmeu) %% 3 clusters
dendrogram(Zavch) %% 3 clusters


Caveu = cophenet(Zaveu,Yeu);

Cavch = cophenet(Zavch,Ych);

Ccmeu = cophenet(Zcmeu,Yeu);





%%% Create Clusters

for i=1:6
    Tcmeu(:,i) = cluster(Zcmeu,'maxclust',i);
end

eva1_cmeu = evalclusters(X,Tcmeu,'CalinskiHarabasz')
eva2_cmeu = evalclusters(X,Tcmeu,'DaviesBouldin')
eva3_cmeu = evalclusters(X,Tcmeu,'silhouette')

plot(eva1_cmeu)
plot(eva2_cmeu)
plot(eva3_cmeu)

figure
NumCluster = 3;
color = Zcmeu(end-NumCluster+2,3)-eps;
[H,T,perm] = dendrogram(Zcmeu, 0, 'colorthreshold', color);

for i=1:6
    Taveu(:,i) = cluster(Zaveu,'maxclust',i);
end

eva1_aveu = evalclusters(X,Taveu,'CalinskiHarabasz')
eva2_aveu = evalclusters(X,Taveu,'DaviesBouldin')
eva3_aveu = evalclusters(X,Taveu,'silhouette')

plot(eva1_aveu)
plot(eva2_aveu)
plot(eva3_aveu)

figure
NumCluster = 3;
color = Zaveu(end-NumCluster+2,3)-eps;
[H,T,perm] = dendrogram(Zaveu, 0, 'colorthreshold', color);

for i=1:6
    Tavch(:,i) = cluster(Zavch,'maxclust',i);
end

eva1_avch = evalclusters(X,Tavch,'CalinskiHarabasz')
eva2_avch = evalclusters(X,Tavch,'DaviesBouldin')
eva3_avch = evalclusters(X,Tavch,'silhouette')

plot(eva1_avch)
plot(eva2_avch)
plot(eva3_avch)

NumCluster = 3;
color = Zavch(end-NumCluster+2,3)-eps;
[H,T,perm] = dendrogram(Zavch, 0, 'colorthreshold', color);
%%% Plot
% silhouette(X,Twaeu2)
% figure
silhouette(X,Tcmeu(:,3))


all_clusters = xlsread('ISB_Project.xlsx', 'All_Clusters');
% ind1 = ward_eu(:,9) == 1;
% ind2 = ward_eu(:,9) == 2;
% cluster1 = ward_eu(ind1,:);
% cluster2 = ward_eu(ind2,:);

%%% For Hierarchical 3 clusters
[p1,tbl1,stats1] = kruskalwallis(all_clusters(:,5),all_clusters(:,9))
[p2,tbl2,stats2] = kruskalwallis(all_clusters(:,6),all_clusters(:,9))
[p3,tbl3,stats3] = kruskalwallis(all_clusters(:,7),all_clusters(:,9))

c1 = multcompare(stats1)
c2 = multcompare(stats2)
c3 = multcompare(stats3)

%%% For Kmeans - FCM - 3 clusters

[p1,tbl1,stats1] = kruskalwallis(all_clusters(:,5),all_clusters(:,13))
[p2,tbl2,stats2] = kruskalwallis(all_clusters(:,6),all_clusters(:,13))
[p3,tbl3,stats3] = kruskalwallis(all_clusters(:,7),all_clusters(:,13))

c1 = multcompare(stats1)
c2 = multcompare(stats2)
c3 = multcompare(stats3)
