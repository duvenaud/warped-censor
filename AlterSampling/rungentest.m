% Gentest run
mkdir('gentestout');
for htest=0:0.1:5
    h = htest;
    gentest
    filename = [num2str(htest), '.png'];
    print(1, ['./gentestout/1_', filename], '-dpng');
    print(2, ['./gentestout/2_', filename], '-dpng');
end