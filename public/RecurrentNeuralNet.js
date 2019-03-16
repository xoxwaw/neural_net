class RecurrentNeuralNet{
    constructor(){
        this.activation = new Activations();
        this.cal = new Calculator();
    }
    rnn_cell_forward(xt, a_prev, parameters){
        /*
        xt: input at time t, matrix of shape (n_x, m)
        a_prev : hidden state at time t-1, matrix of shape (n_a, m)
        parameters:
            Wax: weight matrix multiplying the input, matrix of shape (n_a,n_x)
            Waa: weight matrix multiplying the hidden state, matrix of shape (n_a, n_a)
            Wya: weight matrix relating to the hidden state of the output, matrix of shape (n_y, n_a)
            ba : bias vector of shape (n_a, 1)
            by: bias from hidden state to output, vector of shape (n_y,1)
        return:
            a_next: next hidden state of shape (n_a,m)
            yt_pred: prediction at time step t, shape (n_y,m)
            cache: tuple of values needed for backward pass, contains (a_next, a_prev, xt, parameters)
        */
        var Wax = parameters["Wax"];
        var Waa = parameters["Waa"];
        var Wya = parameters["Wya"];
        var ba = parameters["ba"];
        var by = parameters["by"];
        var a_next = this.activation.tanh(
            this.cal.addMatrixVector(
                this.cal.add(
                    this.cal.dot(Wax, this.cal.transpose(xt)),
                    this.cal.dot(Waa, this.cal.transpose(a_prev)), "+"
                ),ba
            ));
        // console.log(a_next);
        var y_pred = this.activation.softmax(
            this.cal.addMatrixVector(
                this.cal.dot(
                    Wya,
                    this.cal.transpose(a_next),
                ),
                by
            )
        );
        var cache = [a_next, a_prev, xt, parameters];
        return {
            "a_next":a_next,
            "y_pred":y_pred,
            "cache": cache
        };
    }
    rnn_forward(x, a0, parameters){
        /*
        x: input data for every time step, shape (n_x, m, T_x)
        a0: initial hidden state, shape (n_a,m)
        paramters:
            Waa: weight matrix multiplying hidden state, shape (n_a,n_a)
            Wax: weight matrix multiplying input, shape (n_a, n_x)
            Wya: weight matrix multiplying to return output, shape (n_y, n_a)
            ba: biase vector of shape (n_a, 1)
            by: biase vector for output for (n_y,1)
        Returns:
            a: hidden states for every time step, (n_a, m, T-x)
            y_pred: predictions for every timestep, numpy shape (n_y, m, T_x)
            caches: tuples of cache (lists of cache, x)
        */
        var n_x = x.length,
            m = x[0].length,
            T_x = x[0][0].length;
        var n_y = parameters["Wya"].length,
            n_a = parameters["Wya"][0].length;
        var a = this.cal.generateZeros([n_a, m, T_x]),
            y_pred = this.cal.generateZeros([n_y, m, T_x]);
        var a_next = a0;
        var caches= [];
        for (var t = 0; t < T_x; t++){
            var res = this.rnn_cell_forward(
                this.cal.getMMinusOneDim(x, t),
                a_next,
                parameters
            );
            a_next = res["a_next"];
            var a_dim = this.cal.shape(a_next);
            a = this.cal._assignVal_(a, a_next,0, a_dim, t);
            var yt_pred = res["y_pred"];
            var cache = res["cache"];
            let yt_dim = this.cal.shape(this.cal.transpose(yt_pred));
            y_pred = this.cal._assignVal_(y_pred, this.cal.transpose(yt_pred), 0, yt_dim, t);
            caches.push(cache);
        }
        caches = [caches,x];
        return {
            "a": a,
            "y_pred": y_pred,
            "caches": caches};
    }
}
