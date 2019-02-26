// var assert = require('assert');
class NeuralNetwork {
    constructor(x, y) {
        this.input = x;
        this.training_size = y.length;
        this.weights1 = this.generateRandomMatrix(3,x[0].length);//weights
        this.biases1 = this.generateRandomMatrix(3,1);
        this.weights2 = this.generateRandomMatrix(2,this.weights1.length);
        this.biases2 = this.generateRandomMatrix(2,1);
        this.weights3 = this.generateRandomMatrix(1,this.weights2.length);
        this.biases3 = this.generateRandomMatrix(1,1);
        this.z_1 = this.generateRandomMatrix(3,1);
        this.a_1 = this.generateRandomMatrix(3,1);
        this.z_2 = this.generateRandomMatrix(2,1);
        this.a_2 = this.generateRandomMatrix(2,1);
        this.z_3 = this.generateRandomMatrix(1,1);
        this.a_3 = this.generateRandomMatrix(1,1);
        this.y = y; //actual output
        this.output = this.generateZeros(y.length); //initialize predicted output
        this.alpha = 0.1; //learning rate
    }
    generateRandomMatrix(row, col) {
        var matrix = [];
        for (var i = 0; i < row; i++) {
            matrix[i] = [];
            for (var j = 0; j < col; j++) {
                matrix[i][j] = Math.random();
            }
        }
        return matrix;
    }
    generateZeros(col) {
        var vector = [];
        for (var i = 0; i < col; i++) {
            vector[i] = 0;
        }
        return vector;
    }
    _sigmoid(x) {
        return 1 / (1 + Math.pow(Math.E, -x))
    }
    sigmoid(m){
        var res = [];
        for (var i = 0; i < m.length; i++){
            res[i] = [];
            for (var j = 0; j < m[0].length;j++){
                res[i][j] = this._sigmoid(m[i][j]);
            }
        }
        return res;
    }
    _sigmoidDerivative(x) {
        return this._sigmoid(x) * (1 - this._sigmoid(x));
    }
    sigmoidDerivative(m){
        var res = [];
        for (var i = 0; i < m.length; i++){
            res[i] = [];
            for (var j = 0; j < m[0].length;j++){
                res[i][j] = this._sigmoidDerivative(m[i][j]);
            }
        }
        return res;
    }
    sumOf(m){
        var res = 0;
        for (var i = 0; i < m.length; i++){
            for (var j = 0; j < m[0].length;j++){
                res += m[i][j];
            }
        }
        return res;
    }
    add(m1,m2,operator){
        var res= [];
        if (m1.length != m2.length){
            console.log(m1.length,m2.length);
            return [];
        }
        for (var i = 0; i < m1.length;i++){
            res[i] = []
            for (var j = 0; j < m1[0].length;j++){
                if (operator == "+") res[i][j] = m1[i][j] + m2[i][j];
                else if (operator == "-") res[i][j] = m1[i][j] - m2[i][j];
            }
        }
        return res;
    }

    scalarOp(m, num, op){
        var res = [];
        for (var i = 0; i < m.length; i++){
            res[i] = [];
            for (var j = 0; j < m[0].length;j++){
                if (op == "*") res[i][j] = m[i][j] * num;
                else if (op == "+") res[i][j] = m[i][j] + num;
                else if (op == "-") res[i][j] = m[i][j] - num;
            }
        }
        return res;
    }
    elemProduct(m1,m2){
        var res = [];
        if (m1.length != m2.length || m1[0].length != m2[0].length){
            console.log(m1.length, m2.length, m1[0].length, m2[0].length);
        }
        for (var i = 0; i < m1.length; i++){
            res[i] = [];
            for (var j = 0; j < m1[0].length; j++){
                res[i][j] = m1[i][j] * m2[i][j];
            }
        }
        return res;
    }
    dot(m1,m2){
        // assert(m1.length == m2.length);

        var res = [];
        for (var i = 0; i < m1.length; i++){
            res[i] = [];
            for (var j = 0; j < m2.length; j++){
                res[i][j] = 0;
                for (var k = 0; k < m1[0].length;k++){
                    res[i][j] += m1[i][k] * m2[j][k];
                }
            }
        }
        return res;
    }

    feedForward(index) {
        this.z_1 = this.add(this.dot(this.weights1, [this.input[index]]), this.biases1, "+");
        this.a_1 = this.sigmoid(this.z_1);
        this.z_2 = this.add(this.dot(this.weights2, nn.transpose(this.a_1)), this.biases2, "+");
        this.a_2 = this.sigmoid(this.z_2);
        this.z_3 = this.add(this.dot(this.weights3, nn.transpose(this.a_2)), this.biases3, "+");
        this.a_3 = this.sigmoid(this.z_3);
        return this.a_3;
    }
    backpropagate(ind) {
        var d_z_3 = [[this.a_3 - this.y[ind]]];
        var d_W_3 = this.scalarOp(this.dot(d_z_3, this.a_2), 1 / this.training_size, "*") ;
        var d_b_3 = this.sumOf(d_z_3) / this.training_size;

        var d_z_2 = this.elemProduct(this.dot(this.transpose(this.weights3), d_z_3), this.sigmoidDerivative(this.z_2));
        var d_W_2 = this.scalarOp(this.dot(d_z_2, this.a_1), 1 / this.training_size, "*");
        var d_b_2 = this.sumOf(d_z_2) / this.training_size;

        var d_z_1 = this.elemProduct(this.dot(this.transpose(this.weights2), this.transpose(d_z_2)), this.sigmoidDerivative(this.z_1));
        var d_W_1 = this.scalarOp(this.dot(d_z_1, this.transpose([this.input[ind]])), 1 / this.training_size, "*");
        var d_b_1 = this.sumOf(d_z_1) / this.training_size;

        this.weights1 = this.add(this.weights1, this.scalarOp(d_W_1, this.alpha, "*"), "-");
        // console.log(this.weights1);
        this.weights2 = this.add(this.weights2, this.scalarOp(d_W_2, this.alpha, "*"), "-");
        this.weights3 = this.add(this.weights3, this.scalarOp(d_W_3, this.alpha, "*"), "-");

        this.biases1 = this.scalarOp(this.biases1, d_b_1, "-");
        this.biases2 = this.scalarOp(this.biases2, d_b_2, "-");
        this.biases3 = this.scalarOp(this.biases3, d_b_3, "-");

    }
    transpose(matrix) {
        var i, j, t = [];
        var w = matrix.length || 0;
        if (matrix[0].length == null){
            for (i = 0; i < w; i++){
                t[i] = [];
                t[i][0] = matrix[i];
            }
            return t;
        }
        var h = matrix[0] instanceof Array ? matrix[0].length : 0;
        if (h === 0 || w === 0) {
            return [];
        }
        for (i = 0; i < h; i++) {
            t[i] = [];
            for (j = 0; j < w; j++) {
                t[i][j] = matrix[j][i];
            }
        }
        return t;
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
