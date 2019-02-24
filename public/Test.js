var X = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[2,1],[3,1],[3,2],[4,1],[4,2],[4,3]];
var y = [0,0,0,0,0,0,1,1,1,1,1,1];
var nn = new NeuralNetwork(X,y);
// console.log(nn.dot([[1],[2],[3]],[[2],[3],[4]]));
for (var i = 0; i < 1000;i++){
    if (i%100 == 0){
        console.log("for iteration #"+ i);
        console.log("Expected output: "+ y);
        console.log("Predicted output: " + nn.feedForward());
    }    // console.log("Loss: " + nn.error[0]);
    nn.train();
}
