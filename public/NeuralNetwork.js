class NeuralNetwork {
    constructor(x, y) {
        this.input = x;
        this.weights1 = this.generateRandomMatrix(x.length,x[0].length);//weights
        this.y = y; //actual output
        this.output = this.generateZeros(y.length); //initialize predicted output
        this.alpha = 0.1; //learning rate
        this.layer1 = [];//hidden layer 1
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
    sigmoid(x) {
        return 1 / (1 + Math.pow(Math.E, -x))
    }
    sigmoidDerivative(x) {
        return 1 * (1 - x);
    }
    dot(m1, m2) {
        var row = m1.length;
        var col = m1[0].length;
        var result = [];
        for (var i = 0; i < row; i++){
            result[i] = 0;
            for (var j = 0; j < col; j++){
                result[i] += m1[i][j] * m2[i][j];
            }
        }
        return result;
    }
    dotVectorMatrix(v,m){
        var res = [];
        for (var i = 0; i < m.length; i++){
            res[i] = [];
            for (var j = 0; j < m[0].length; j++){
                res[i][j]= m[i][j] * v[i];
            }
        }
        return res;
    }
    add(m1,m2){
        var result = [];
        var row = m1.length;
        var col = m1[0].length;
        for (var i = 0; i < row; i++){
            result[i] = [];
            for (var j = 0 ; j < col; j ++){
                result[i][j] = m1[i][j] + m2[i][j];
            }

        }
        return result;
    }
    substractVector(v1,v2){
        var result = [];
        var col = v1.length;
        for (var i = 0; i < col; i++){
            result[i] = v1[i] - v2[i];
        }
        return result;
    }
    substractMatrix(m1,m2){
        var res = [];
        for (var i = 0; i < m1.length; i++){
            res[i] = [];
            for (var j = 0; j < m1[0].length; j++){
                res[i][j] = m1[i][j] - m2[i][j];
            }
        }
        return res;
    }
    multiplyVector(v, num){
        var new_v = [];
        for (var i = 0; i < v.length; i++){
            new_v[i] = num * v[i];
        }
        return new_v;
    }
    multiplyMatrix(m, num){
        new_m = [];
        for (var i = 0; i < m.length; i++){
            new_m[i] = [];
            for (var j = 0; j < m.length; j++){
                new_m[i][j] = m[i][j] * num;
            }
        }
        return new_m
    }
    feedForward() {
        var forward = this.dot(this.input, this.weights1); //z = x.W
        for (var i = 0; i < this.y.length; i++){
            this.layer1[i] = this.sigmoid(forward[i]); //a[1] = sigmoid(z)
        }
        this.output = this.layer1;
        return this.output;
    }
    backpropagate() {
        var error = this.multiplyVector(this.substractVector(this.y, this.output),-2);//error = -2(y - (x.W))
        var d_weight = this.dotVectorMatrix(error, this.input);// dE/dW = x . error
        this.weights1 = this.substractMatrix(this.weights1,d_weight);// modify weight

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
    train() {
        this.feedForward();//
        this.backpropagate();
    }
}
