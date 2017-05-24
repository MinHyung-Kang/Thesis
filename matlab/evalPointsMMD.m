function [HInfo]= evalPointsMMD(points, points_nuts)
    points_total = [points;points_nuts];
    labels = [ones(1,size(points_nuts,1)) ones(1,size(points,1)) * -1];
    [~,HInfo]= mmd(points_total, labels);
end
