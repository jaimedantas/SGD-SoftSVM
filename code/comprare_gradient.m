% -- Losses

dot_product = -4:0.01:0.99; % t〈w,x〉< 1
[a,D] = size(dot_product);

% tx
constant = 1;

% logistic gradient
logistic_i = [];
for i = 1:D
    logistic_i(i) = -1 * constant /(exp(dot_product(i)) + 1);
end
figure
l = plot(dot_product, logistic_i,'linewidth',2);
hold on 

% hinge loss
hinge_i = [];
for i = 1:D
    hinge_i(i) = -1 * constant;
end
h = plot(dot_product, hinge_i,'linewidth',2);
hold on

% quad loss
quad_i = [];
for i = 1:D
    quad_i(i) = -2 * constant;
end
q = plot(dot_product, quad_i,'linewidth',2);
hold on

% exp loss
exponential_i = [];
for i = 1:D
    exponential_i(i) = -1 * constant /(exp(dot_product(i)));
end
l = plot(dot_product, exponential_i,'linewidth',2);
hold on 

% plot
grid;
ylim([-2 2])
legend( 'Logistic', 'Hinge', 'Quad', 'Exponential')
ylabel('Losses'); 
xlabel('t<x,w>'); 
title('Empirical Losses t〈w,x〉<1');
hold off
print -depsc loss4comp
%--------- other side
dot_product = 1:0.01:4; % t〈w,x〉>= 1
[a,D] = size(dot_product);

% tx
constant = 1;

% logistic gradient
logistic_i = [];
for i = 1:D
    logistic_i(i) = -1 * constant /(exp(dot_product(i)) + 1);
end
figure
l = plot(dot_product, logistic_i,'linewidth',2);
hold on 

% hinge loss
hinge_i = [];
for i = 1:D
    hinge_i(i) = 0;
end
h = plot(dot_product, hinge_i,'linewidth',2);
hold on

% quad loss
quad_i = [];
for i = 1:D
    quad_i(i) = 0;
end
q = plot(dot_product, quad_i,'linewidth',2);
hold on

% exp loss
exponential_i = [];
for i = 1:D
    exponential_i(i) = -1 * constant /(exp(dot_product(i)));
end
l = plot(dot_product, exponential_i,'linewidth',2);
hold on 

% plot
grid;
ylim([-2 2])
legend( 'Logistic', 'Hinge', 'Quad', 'Exponential')
ylabel('Losses'); 
xlabel('t<x,w>'); 
title('Empirical Losses t〈w,x〉≥1');
hold off
print -depsc loss4comp