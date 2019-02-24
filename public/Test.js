var X = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[2,1],[3,1],[3,2],[4,1],[4,2],[4,3]];
// var X = [[1],[2],[3],[4],[5],[6]];
// var y = [1,4,9,16,25,36];
var y = [0,0,0,0,0,0,1,1,1,1,1,1];
var nn = new NeuralNetwork(X,y);
// console.log(nn.dot([[1],[2],[3]],[[2],[3],[4]]));
for (var i = 0; i < 10000;i++){
    if (i%1000 == 0){
        console.log("for iteration #"+ i);
        console.log("Expected output: "+ y);
        console.log("Predicted output: " + nn.feedForward());
    }    // console.log("Loss: " + nn.error[0]);
    nn.train();
}
var X_test = [[1,2],[5,6],[5,4],[-1,-2],[100,101],[1000,999],[7,9],[10,90],[100,1]];
// var X_test = [[3],[7],[100],[-1],[9],[0]];
console.log(nn.predict(X_test));
