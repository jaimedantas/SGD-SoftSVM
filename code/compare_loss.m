% -- Losses

dot_product = -4:0.001:4;
[a,D] = size(dot_product);

% logistic loss
for i = 1:D
    logistic_i(i) = log(1 + exp(-1 * dot_product(i)));
end
figure
l = plot(dot_product, logistic_i,'linewidth',2);
hold on 

% binary loss
for i = 1:D
    if dot_product(i) <= 0
        binary_i(i) = 1;
    else
        binary_i(i) = 0;
    end
end
b = plot(dot_product, binary_i,'linewidth',2);
hold on

% hinge loss
for i = 1:D
    hinge_i(i) = max(0,1 - dot_product(i));
end
h = plot(dot_product, hinge_i,'linewidth',2);
hold on

% quad loss
for i = 1:D
    quad_i(i) = (max(0,1 - dot_product(i)))^2;
end
q = plot(dot_product, quad_i,'linewidth',2);
hold on

% exp loss
for i = 1:D
    exponential_i(i) = exp(-1 * dot_product(i));
end
l = plot(dot_product, exponential_i,'linewidth',2);
hold on 

% plot
grid;
%ylim([0 4])
legend( 'Logistic', 'Binary', 'Hinge', 'Quad', 'Exponential')
ylabel('Losses'); 
xlabel('t<x,w>'); 
title('Empirical Losses');
hold off
print -depsc loss3comp
