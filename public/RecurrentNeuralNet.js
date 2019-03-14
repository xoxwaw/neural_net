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
    lstm_cell_forward(xt, c_prev, a_prev, parameters){
        /*
        xt - input data at timestep t, shape of (n_x, m)
        c_prev - memory state at timestep t-1, shape of (n_a, m)
        a_prev - Hidden state of timetep t-1, shape of (n_a, m)
        parameters:
            Wf - weight for the forget gate, shape of (n_a, n_a + n_x)
            bf - bias for forget gate, shape of (n_a, 1)
            Wi - weight for save gate, shape of (n_a, n_a + n_x)
            bi - bias for save gate, shape of (n_a, 1)
            Wc - weight for matrix of the first tanh, shape of (n_a, n_a + n_x)
            bc - bias for the first tanh, shape of (n_a, 1)
            Wo - weight for focus gate, shape of (n_a, n_a + n_x)
            bo - bias for focus gate, shape of (n_a, 1)
            Wy - weight related for the output layer, shape of (n_y, n_a)
            by - bias related for the output layer, shape of (n_y, 1)
        Returns:
            a_next : the next hidden state, shape of (n_a, m)
            c_next : the next memory state, shape of (n_a, m)
            yt_pred: prediction for the timestep, shape of (n_y, n_a)
            cache: tuples of values needed for the backpass,
                contains (a_next, c_next, a_prev, c_prev, xt, parameters)

        Note: ft/it/ot stand for the forget/update/output gates, cct stands for
            the candidate value (c tilda), c stands for the memory value
        */
        var Wf = parameters["Wf"];
        var bf = parameters["bf"];
        var Wi = parameters["Wi"];
        var bi = parameters["bi"];
        var Wc = parameters["Wc"];
        var bc = parameters["bc"];
        var Wo = parameters["Wo"];
        var bo = parameters["bo"];
        var Wy = parameters["Wy"];
        var by = parameters["by"];

        var n_x = this.cal.shape(xt)[0];
        var m = this.cal.shape(xt)[1];
        var n_y = this.cal.shape(Wy)[0];
        var n_a = this.cal.shape(Wy)[1];

        var concat = this.cal.concatenateRow(a_prev, xt)
        var ft = this.activation.sigmoid(this.cal.addMatrixVector(
            this.cal.dot(
                Wf, this.cal.transpose(concat)
            ),bf
        ));
        var it = this.activation.sigmoid(this.cal.addMatrixVector(
            this.cal.dot(
                Wi, this.cal.transpose(concat)
            ), bi
        ));
        var cct = this.activation.tanh(this.cal.addMatrixVector(
            this.cal.dot(
                Wc, this.cal.transpose(concat)
            ), bc
        ));
        var c_next = this.cal.add(
            this.cal.elemProduct(ft, c_prev),
            this.cal.elemProduct(it, cct), "+"
        );
        var ot = this.activation.sigmoid(this.cal.addMatrixVector(
            this.cal.dot(
                Wo, this.cal.transpose(concat)
            ), bo
        ));
        var a_next = this.cal.elemProduct(
            ot, this.activation.tanh(c_next)
        );
        var yt_pred = this.activation.softmax(
            this.cal.addMatrixVector(
                this.cal.dot(
                    Wy, this.cal.transpose(a_next)
                ), by
            )
        );
        console.log(this.cal.shape(a_prev));
        var cache = [a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters];
        return {
            "a_next" : a_next,
            "c_next" : c_next,
            "yt_pred": yt_pred,
            "cache"  : cache
        };
    }
    lstm_forward(x, a0, parameters){
        /*
        x - input data for all timestep, (n_x, m, T_x)
        a0 - hidden state (n_a, m)
        parameters:
            Wf - weight for the forget gate, shape of (n_a, n_a + n_x)
            bf - bias for forget gate, shape of (n_a, 1)
            Wi - weight for save gate, shape of (n_a, n_a + n_x)
            bi - bias for save gate, shape of (n_a, 1)
            Wc - weight for matrix of the first tanh, shape of (n_a, n_a + n_x)
            bc - bias for the first tanh, shape of (n_a, 1)
            Wo - weight for focus gate, shape of (n_a, n_a + n_x)
            bo - bias for focus gate, shape of (n_a, 1)
            Wy - weight related for the output layer, shape of (n_y, n_a)
            by - bias related for the output layer, shape of (n_y, 1)

        TO RETURN:
            a - hidden state for every timestep (n_a, m, T_x)
            y - predictions for all timestep (n_y, m, T_x)
            caches - values needed for the backward pass
        */
        var caches = [];

        var x_shape = this.cal.shape(x);
        var m = x_shape[1];
        var T_x = x_shape[2];

        var y_shape = this.cal.shape(parameters["Wy"]);
        var n_y = y_shape[0];
        var n_a = y_shape[1];

        var a = this.cal.generateZeros([n_a, m, T_x]);
        var c = this.cal.generateZeros([n_a, m, T_x]);
        var y_pred = this.cal.generateZeros([n_y, m, T_x]);

        var a_next = a0;
        var c_next = this.cal.generateZeros(this.cal.shape(a_next));
        for (var t = 0; t < T_x; t++){
            var res = this.lstm_cell_forward(
                this.cal.getMMinusOneDim(x,t),
                a_next, c_next, parameters
            );
            a_next = res["a_next"];
            var a_dim = this.cal.shape(a_next);
            a = this.cal._assignVal_(a, a_next,0, a_dim, t);
            var yt_pred = res["yt_pred"];
            var cache = res["cache"];
            let yt_dim = this.cal.shape(this.cal.transpose(yt_pred));
            y_pred = this.cal._assignVal_(y_pred, this.cal.transpose(yt_pred), 0, yt_dim, t);
            c_next = res["c_next"];
            var c_dim = this.cal.shape(c_next);
            c = this.cal._assignVal_(c, c_next, 0, c_dim, t);
            caches.push(cache);
        }
        caches = [caches, x];

        return {
            "a": a,
            "y_pred": y_pred,
            "caches": caches};

    }
    rnn_cell_backward(da_next, cache){
        /*Single time step
        da_next : - gradient of loss wrt hidden state
        cache - output from rnn_cell_forward

        Returns:
        Gradients:
            dx - gradient of input date, shape (n_x, m)
            da_prev - gradient of previous hidden state, shape (n_a, m)
            dWax - gradient of input-to-hidden
        */

    }
    rnn_backward(){

    }
    lstm_cell_backward(da_next, dc_next, caches){
        /*single timestep
        da_next - gradient of hidden state, shape (n_a, m)
        dc_next - gradient of memory state, shape (n_a, m)
        cache - cache store information for the forward pass

        To RETURN: gradients containing:
            dxt - gradient of input data at timestep t (n_x, m)
            da_prev - gradient of the previous hidden state (n_a, m)
            dc_prev - gradient of the previous memory state (n_a, m, T_x)
            dWf - gradient wrt the weight of forget state, shape (n_a, n_a + n_x)
            dWi - gradient wrt the weight of the input state, (n_a, n_a + n_x)
            dWc - gradient wrt the weight of the memory state, (n_a, n_a + n_x)
            dWo - gradient wrt the weight of the save state, (n_a, n_a + n_x)
            dbf - gradient wrt the bias of forget state (n_a, 1)
            dbi - gradient wrt the bias of the input state (n_a, 1)
            dbc - gradient wrt the bias of the memory state (n_a, 1)
            dbo - gradient wrt the bias of the save state, (n_a, 1)
        */
        var a_next = caches[0],
            c_next = caches[1],
            a_prev = caches[2],
            c_prev = caches[3],
            ft = caches[4],
            it = caches[5],
            cct= caches[6],
            ot = caches[7],
            xt = caches[8],
            parameters = caches[9];
        var da_shape = this.cal.shape(da_next);
        var n_x = da_shape[0],
            m   = da_shape[1],
            n_a = this.cal.shape(a_next)[0];

        var dot = this.cal.elemProduct(
            da_next, this.cal.elemProduct(
                this.activation.tanh(c_next),
                this.cal.elemProduct(
                    ot, this.cal.scalarOpRe(1, ot, "-")
                )
            )
        );

        var dcct = this.cal.elemProduct(
            this.cal.add(
                this.cal.elemProduct(
                    da_next, this.cal.elemProduct(
                        ot, this.cal.scalarOpRe(
                            1, this.cal.scalarOp(
                                this.activation.tanh(c_next),2,
                                "**"
                                ), "-"
                            )
                        )
                    ),dc_next, "+"
            ), this.cal.elemProduct(
                it, this.cal.scalarOpRe(
                    1, this.cal.scalarOp(cct, 2, "**"), "-"
                )
            )
        );
        // console.log(dcct);
        var dit = this.cal.elemProduct(
            this.cal.add(
               this.cal.elemProduct(
                   da_next, this.cal.elemProduct(
                       ot, this.cal.scalarOpRe(
                           1, this.cal.scalarOp(
                               this.activation.tanh(c_next), 2, "**"
                           ), "-"
                       )
                   )
               ), dc_next, "+"
           ), this.cal.elemProduct(
               cct, this.cal.elemProduct(
                   this.cal.scalarOpRe(1, it, "-"), it
               )
           )
       );

       var dft = this.cal.elemProduct(
           this.cal.add(
               this.cal.elemProduct(
                   da_next, this.cal.elemProduct(
                       ot, this.cal.scalarOpRe(
                           1, this.cal.scalarOp(this.activation.tanh(c_next), 2, "**"),"-"
                       )
                   )
               ), dc_next, "+"
           ), this.cal.elemProduct(
               c_prev, this.cal.elemProduct(
                   ft, this.cal.scalarOpRe(1, ft, "-")
               )
           )
       );

       var dWf = this.cal.dot(
           dft, this.cal.concatenateRow(
               (a_prev),
               (xt)
           )
       );

       var dWi = this.cal.dot(
           dit, this.cal.concatenateRow(
               (a_prev),
               (xt)
           )
       );

       var dWc = this.cal.dot(
           dcct, this.cal.concatenateRow(
               (a_prev),
               (xt)
           )
       );

       var dWo = this.cal.dot(
           dot, this.cal.concatenateRow(
               (a_prev),
               (xt)
           )
       );
       var dbf = this.cal.sumHorizontal(dft);
       var dbi = this.cal.sumHorizontal(dit);
       var dbo = this.cal.sumHorizontal(dot);
       var dbc = this.cal.sumHorizontal(dcct);

       var Wf = parameters["Wf"],
            Wo= parameters["Wo"],
            Wi= parameters["Wi"],
            Wc= parameters["Wc"];

       var da_prev = this.cal.add(
           this.cal.dot(
               this.cal.transpose(this.cal.sliceMatrix(Wf, n_a, Wf[0].length)),
               this.cal.transpose(dft)
           ),this.cal.add(
               this.cal.dot(
                   this.cal.transpose(this.cal.sliceMatrix(Wc, n_a, Wc[0].length)),
                   this.cal.transpose(dcct)
               ),this.cal.add(
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(Wi, n_a, Wi[0].length)),
                       this.cal.transpose(dit)
                   ),
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(Wo, n_a, Wo[0].length)),
                       this.cal.transpose(dot)
                   ), "+"
               ), "+"
           ), "+"
       );

       var dc_prev = this.cal.elemProduct(
           this.cal.add(
               this.cal.elemProduct(
                   da_next,this.cal.elemProduct(
                       ot, this.cal.scalarOpRe(
                           1, this.cal.scalarOp(this.activation.tanh(c_next),2, "**"), "-"
                       )
                   )
               ), dc_next, "+"
           ), ft
       );
       console.log(this.cal.shape(this.cal.sliceMatrix(Wf,n_a,Wf[0].length)), this.cal.shape(dft));
       var dxt = this.cal.add(
           this.cal.dot(
               this.cal.transpose(this.cal.sliceMatrix(Wf, n_a, Wf[0].length)),
               this.cal.transpose(dft)
           ), this.cal.add(
               this.cal.dot(
                   this.cal.transpose(this.cal.sliceMatrix(Wc, n_a, Wc[0].length)),
                   this.cal.transpose(dcct)
               ),this.cal.add(
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(Wo, n_a, Wo[0].length)),
                       this.cal.transpose(dot)
                   ),
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(Wi, n_a, Wi[0].length)),
                       this.cal.transpose(dit)
                   ), "+"
               ), "+"
           ), "+"
       )
       return {
           "dxt": dxt,
           "da_prev" : da_prev,
           "dc_prev" : dc_prev,
           "dWf" : dWf,
           "dbf" : dbf,
           "dWi" : dWi,
           "dbi" : dbi,
           "dWc" : dWc,
           "dbc" : dbc,
           "dWo" : dWo,
           "dbo" : dbo
       }
    }
    lstm_backward(da, cache){
        /*
        da - gradient wrt hidden state (n_a, m, T_x)
        cache - info for forward pass

        To RETURN: gradients containing:
            dx - gradient of input data (n_x, m, T_x)
            da0 - gradient of the first hidden state (n_a, m)
            dWf - gradient wrt the weight of forget state, shape (n_a, n_a + n_x)
            dWi - gradient wrt the weight of the input state, (n_a, n_a + n_x)
            dWc - gradient wrt the weight of the memory state, (n_a, n_a + n_x)
            dWo - gradient wrt the weight of the save state, (n_a, n_a + n_x)
            dbf - gradient wrt the bias of forget state (n_a, 1)
            dbi - gradient wrt the bias of the input state (n_a, 1)
            dbc - gradient wrt the bias of the memory state (n_a, 1)
            dbo - gradient wrt the bias of the save state, (n_a, 1)
        */
        var a1 = caches[0],
            c1 = caches[1],
            a0 = caches[2],
            c0 = caches[3],
            f1 = caches[4],
            i1 = caches[5],
            cc1= caches[6],
            o1 = caches[7],
            x1 = caches[8],
            parameters = caches[9];
    }
}
