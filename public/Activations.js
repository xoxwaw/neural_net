class Activations {
    constructor() {
        this.cal = new Calculator();
    }

    tanh(m) {
        var res = [];
        for (var i = 0; i < m.length; i++) {
            res[i] = [];
            for (var j = 0; j < m[0].length; j++) {
                res[i][j] = Math.tanh(m[i][j]);
            }
        }
        return res;
    }
    _sigmoid(x) {
        return 1 / (1 + Math.pow(Math.E, -x))
    }
    sigmoid(m) {
        var res = [];
        for (var i = 0; i < m.length; i++) {
            res[i] = [];
            for (var j = 0; j < m[0].length; j++) {
                res[i][j] = this._sigmoid(m[i][j]);
            }
        }
        return res;
    }
    _sigmoidDerivative(x) {
        return this._sigmoid(x) * (1 - this._sigmoid(x));
    }
    sigmoidDerivative(m) {
        var res = [];
        for (var i = 0; i < m.length; i++) {
            res[i] = [];
            for (var j = 0; j < m[0].length; j++) {
                res[i][j] = this._sigmoidDerivative(m[i][j]);
            }
        }
        return res;
    }
    _softmax(arr) {
        return arr.map(function(value, index) {
            return Math.exp(value) / arr.map(function(y /*value*/ ) {
                return Math.exp(y)
            }).reduce(function(a, b) {
                return a + b
            })
        })
    }
    softmax(m) {
        var new_m = this.cal.transpose(m);
        var to_return = [];
        for (var i = 0; i < new_m.length; i++){
            to_return.push(this._softmax(new_m[i]));
        }
        return to_return;
    }
}
