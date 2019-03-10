class Calculator{
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
    _generateEmpty(arr,dim,i){
        if (i < dim.length){
            for (var j = 0; j < dim[i]; j++){
                arr[j] = [];
            }
            for (var k = 0; k < arr.length; k++){
                this._generateEmpty(arr[k],dim,i+1)
            }
        }
        return arr
    }
    _generateZeros(arr){
        if (arr[0][0] != null){
            for (var i = 0; i< arr.length; i++){
                this._generateZeros(arr[i]);
            }
        }else{
            for (var i = 0; i < arr.length; i++){
                arr[i] = 0;
            }
        }
        return arr
    }
    _generateRandom(arr){
        if (arr[0][0] != null){
            for (var i = 0; i< arr.length; i++){
                this._generateRandom(arr[i]);
            }
        }else{
            for (var i = 0; i < arr.length; i++){
                arr[i] = Math.random();
            }
        }
        return arr;
    }
    generateZeros(dim) {
        var arr = this._generateEmpty([],dim,0);
        return this._generateZeros(arr);
    }
    generateRandom(dim){
        var arr = this._generateEmpty([],dim,0);
        return this._generateRandom(arr);
    }
    _assignVal_(arr,new_arr, i, dim,t){
        if (i < dim.length-1){
            for (var j = 0; j< dim[i]; j++){
                this._assignVal_(arr[j],new_arr[j],i+1, dim,t);
            }
        }else{
            for (var j = 0; j < new_arr.length;j++){
                arr[j][t]= new_arr[j] ;
            }
        }
        return arr;
    }
    _assignVal(arr,new_arr, i, dim,t){
        if (i < dim.length-1){
            for (var j = 0; j< dim[i]; j++){
                this._assignVal(arr[j],new_arr[j],i+1, dim,t);
            }
        }else{
            for (var j = 0; j < new_arr.length;j++){
                new_arr[j] = arr[j][t];
            }
        }
        return new_arr;
    }
    _shape(arr,dim){
        if (arr[0] != null){
            dim.push(arr.length);
            this._shape(arr[0],dim);
        }
        return dim
    }
    shape(arr){
        return this._shape(arr,[]);
    }
    getMMinusOneDim(arr,t){
        var dim = this.shape(arr);
        var new_dim = dim.slice(0,dim.length-1);
        var new_arr = this.generateZeros(new_dim);
        new_arr = this._assignVal(arr, new_arr, 0, new_dim,t);
        return new_arr;
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

    concatenateCol(m1, m2){
        for (var i = 0; i < m1.length;i++){
            for (var j = 0; j < m2[0].length; j++){
                m1[i].push(m2[i][j]);
            }
        }
        return m1;
    }

    concatenateRow(m1,m2){
        for (var i = 0 ; i < m2.length; i++){
            m1.push(m2[i]);
        }
        return m1;
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
    addMatrixVector(m,v){
        var res = [];
        for (var i = 0; i< m.length;i++){
            res[i] = [];
            for (var j = 0; j < m[0].length;j++){
                res[i][j] = parseFloat(m[i][j]) + parseFloat(v[i]);
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

}
