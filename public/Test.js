const activation = new Activations();
const cal = new Calculator();
var rnn = new RecurrentNeuralNet();
function testNeuralNet(){
    var X = [[1,2,3],[2,3,4],[3,4,5],[6,7,8],[8,9,10],[10,13,15],[2,3,1],[5,3,1],[6,3,2],[6,4,1],[8,4,2],[7,4,3]];
    var y = [0,0,0,0,0,0,1,1,1,1,1,1];
    // var X = [[1,0],[2,1],[3,8],[-1,0],[-2,1],[-3,7],[4,15],[-4,10],[1,2],[2,5],[3,10],[-2,6],[-3,12],[0,1],[5,27],[9,84]];
    // var y = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1];
    var nn = new NeuralNetwork(X,y);
    for (var i = 0; i < 100000;i++){
        nn.train();
    }
    // var X_test = [[2,0],[2,9],[-3,-2],[7,48],[5,24],[1,1],[2,4]];
    var X_test = [[4,2,1],[4,5,6],[6,5,4],[9,-1,-2],[100,101,1000],[1000,999,998],[7,9,11],[10,90,900],[100,1,0]];
    console.log(nn.predict(X_test));
}
function testRecurrentNetCell(){

    var xt = cal.generateRandomMatrix(3,10);
    var a_prev = cal.generateRandomMatrix(5,10);
    var Waa = cal.generateRandomMatrix(5,5);
    var Wax = cal.generateRandomMatrix(5,3);
    var Wya = cal.generateRandomMatrix(2,5);
    var ba = cal.generateRandomMatrix(5,1);
    var by = cal.generateRandomMatrix(2,1);
    var parameters = {
        "Waa" : Waa,
        "Wax" : Wax,
        "Wya" : Wya,
        "ba"  : ba,
        "by"  : by
    };
    var getResults = rnn.rnn_cell_forward(xt, a_prev, parameters);
    var a_next = getResults["a_next"];
    var y_pred = getResults["y_pred"];
    var cache = getResults["cache"];
    console.log(a_next);
    console.log(y_pred);
    console.log(cache);
}
function testRecurrentNet(){
    var x = cal.generateRandom([3,10,4]);
    var a0 = cal.generateRandom([5,10]);
    var Waa = cal.generateRandom([5,5]);
    var Wax = cal.generateRandom([5,3]);
    var Wya = cal.generateRandom([2,5]);
    var ba = cal.generateRandom([5,1]);
    var by = cal.generateRandom([2,1]);
    var parameters = {
        "Waa" : Waa,
        "Wax" : Wax,
        "Wya" : Wya,
        "ba" : ba,
        "by" : by
    }
    var res = rnn.rnn_forward(x,a0, parameters);
    var a = res["a"];
    var y_pred = res["y_pred"];
    var caches = res["caches"];
    console.log(a[4][1]);
    console.log(cal.shape(a));
    console.log(y_pred[1][3]);
    console.log(cal.shape(y_pred));
    console.log(caches[1][1][3]);
    console.log(caches.length);
}
testRecurrentNet();
