function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

positiv=find(y==1);
negativ=find(y==0);
plot(X(positiv,1),X(positiv,2),'k+');
plot(X(negativ,1),X(negativ,2),'ko');


hold off







% =========================================================================



hold off;

end
