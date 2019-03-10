// var assert = require('assert');
class NeuralNetwork {
    constructor(x, y) {
        this.calculator = new Calculator();
        this.activation = new Activations();
        this.input = x;
        this.training_size = y.length;
        this.weights1 = this.calculator.generateRandomMatrix(3,x[0].length);//weights
        this.biases1 = this.calculator.generateRandomMatrix(3,1);
        this.weights2 = this.calculator.generateRandomMatrix(2,this.weights1.length);
        this.biases2 = this.calculator.generateRandomMatrix(2,1);
        this.weights3 = this.calculator.generateRandomMatrix(1,this.weights2.length);
        this.biases3 = this.calculator.generateRandomMatrix(1,1);
        this.z_1 = this.calculator.generateRandomMatrix(3,1);
        this.a_1 = this.calculator.generateRandomMatrix(3,1);
        this.z_2 = this.calculator.generateRandomMatrix(2,1);
        this.a_2 = this.calculator.generateRandomMatrix(2,1);
        this.z_3 = this.calculator.generateRandomMatrix(1,1);
        this.a_3 = this.calculator.generateRandomMatrix(1,1);
        this.y = y; //actual output
        // this.output = this.calculator.generateZeros(y.length); //initialize predicted output
        this.alpha = 0.1; //learning rate
    }
    
    feedForward(index) {
        this.z_1 = this.calculator.add(this.calculator.dot(this.weights1, [this.input[index]]), this.biases1, "+");
        this.a_1 = this.activation.sigmoid(this.z_1);
        this.z_2 = this.calculator.add(this.calculator.dot(this.weights2, nn.transpose(this.a_1)), this.biases2, "+");
        this.a_2 = this.activation.sigmoid(this.z_2);
        this.z_3 = this.calculator.add(this.calculator.dot(this.weights3, nn.transpose(this.a_2)), this.biases3, "+");
        this.a_3 = this.activation.sigmoid(this.z_3);
        return this.a_3;
    }
    backpropagate(ind) {
        var d_z_3 = [[this.a_3 - this.y[ind]]];
        var d_W_3 = this.calculator.scalarOp(this.calculator.dot(d_z_3, this.a_2), 1 / this.training_size, "*") ;
        var d_b_3 = this.sumOf(d_z_3) / this.training_size;

        var d_z_2 = this.calculator.elemProduct(this.calculator.dot(this.calculator.transpose(this.weights3), d_z_3), this.activation.sigmoidDerivative(this.z_2));
        var d_W_2 = this.calculator.scalarOp(this.calculator.dot(d_z_2, this.a_1), 1 / this.training_size, "*");
        var d_b_2 = this.sumOf(d_z_2) / this.training_size;

        var d_z_1 = this.calculator.elemProduct(this.calculator.dot(this.calculator.transpose(this.weights2), this.calculator.transpose(d_z_2)), this.activation.sigmoidDerivative(this.z_1));
        var d_W_1 = this.calculator.scalarOp(this.calculator.dot(d_z_1, this.calculator.transpose([this.input[ind]])), 1 / this.training_size, "*");
        var d_b_1 = this.sumOf(d_z_1) / this.training_size;

        this.weights1 = this.calculator.add(this.weights1, this.calculator.scalarOp(d_W_1, this.alpha, "*"), "-");
        this.weights2 = this.calculator.add(this.weights2, this.calculator.scalarOp(d_W_2, this.alpha, "*"), "-");
        this.weights3 = this.calculator.add(this.weights3, this.calculator.scalarOp(d_W_3, this.alpha, "*"), "-");

        this.biases1 = this.calculator.scalarOp(this.biases1, d_b_1, "-");
        this.biases2 = this.calculator.scalarOp(this.biases2, d_b_2, "-");
        this.biases3 = this.calculator.scalarOp(this.biases3, d_b_3, "-");

    }

    predict(x){
        this.input = x;
        var res = [];
        for (var i = 0; i< x.length; i++){
            res.push(this.feedForward(i)[0][0]);
        }
        return res;

    }
    train() {
        for (var i = 0; i < this.training_size; i++){
            this.feedForward(i);//
            this.backpropagate(i);
        }

    }
}
