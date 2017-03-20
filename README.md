# Linear-classification w/o using MATLAB data science toolbox

Linear classification has been implemented for whitespace delimited data with last collumn representing class  
Classification is done using logistic regression with iterative least squares.  
this function classifies into two classes only  

if the class is 1 the class remains 1  
if the class is not 1 the class is changed to 0  



place the contents of the folder in the mounted MATLAB folder  
The MATLAB script has been implemented as a function  
  
TO RUN THE CODE  
IN THE COMMAND WINDOW INSIDE MATLAB  

logistic_regression('training_file',degree,'test_file')

example
logistic_regression('pendigits_training.txt',1,'pendigits_test.txt')  

logistic_regression('pendigits_training.txt',2,'pendigits_test.txt')  

DO NOT RUN AS  
logistic_regression <training_file> <degree> <test_file>  
example of incorrect call  
logistic_regression pendigits_training.txt 2 pendigits_test.txt  
