class LSTM{
    constructor(X,y,n_a){
        this.cal = new Calculator();
        this.activation = new Activations();
        this.X = X;
        var dim_X = this.cal.shape(X);
        this.n_x = dim_X[0];
        this.m = dim_X[1];
        this.T_x = dim_X[2];
        this.n_a = n_a;
        this.y = y;
        var dim_y = this.cal.shape(y);
        this.n_y = dim_y[0];
        this.a0 = cal.generateRandom([this.n_a,this.m]);
        this.Wf = this.cal.generateRandom([n_a, n_a + this.n_x]);
        this.Wi = this.cal.generateRandom([n_a, n_a + this.n_x]);
        this.Wc = this.cal.generateRandom([n_a, n_a + this.n_x]);
        this.Wo = this.cal.generateRandom([n_a, n_a + this.n_x]);
        this.bf = this.cal.generateRandom([n_a,1]);
        this.bi = this.cal.generateRandom([n_a,1]);
        this.bc = this.cal.generateRandom([n_a,1]);
        this.bo = this.cal.generateRandom([n_a,1]);
        this.Wy = this.cal.generateRandom([this.n_y, this.n_a]);
        this.by = this.cal.generateRandom([this.n_y],1);
        this.a = this.cal.generateRandom([this.n_a, this.m, this.T_x]);
        this.c = this.cal.generateRandom([this.n_a, this.m, this.T_x]);
        this.da = cal.generateRandom([n_a,this.m,this.T_x]);
        this.y_pred = this.cal.generateZeros([this.n_y, this.m, this.T_x]);
    }
    lstm_cell_forward(xt, c_prev, a_prev){
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
        var concat = this.cal.concatenateRow(a_prev, xt)
        var ft = this.activation.sigmoid(this.cal.addMatrixVector(
            this.cal.dot(
                this.Wf, this.cal.transpose(concat)
            ),this.bf
        ));
        var it = this.activation.sigmoid(this.cal.addMatrixVector(
            this.cal.dot(
                this.Wi, this.cal.transpose(concat)
            ), this.bi
        ));
        var cct = this.activation.tanh(this.cal.addMatrixVector(
            this.cal.dot(
                this.Wc, this.cal.transpose(concat)
            ), this.bc
        ));
        var c_next = this.cal.add(
            this.cal.elemProduct(ft, c_prev),
            this.cal.elemProduct(it, cct), "+"
        );
        var ot = this.activation.sigmoid(this.cal.addMatrixVector(
            this.cal.dot(
                this.Wo, this.cal.transpose(concat)
            ), this.bo
        ));
        var a_next = this.cal.elemProduct(
            ot, this.activation.tanh(c_next)
        );
        var yt_pred = this.activation.softmax(
            this.cal.addMatrixVector(
                this.cal.dot(
                    this.Wy, this.cal.transpose(a_next)
                ), this.by
            )
        );
        var cache = [a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt];
        return {
            "a_next" : a_next,
            "c_next" : c_next,
            "yt_pred": yt_pred,
            "cache"  : cache
        };
    }
    lstm_forward(){
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

        var a_next = this.a0;
        var c_next = this.cal.generateZeros(this.cal.shape(a_next));
        for (var t = 0; t < this.T_x; t++){
            var res = this.lstm_cell_forward(
                this.cal.getMMinusOneDim(this.X,t),
                a_next, c_next
            );
            a_next = res["a_next"];
            var a_dim = this.cal.shape(a_next);
            this.a = this.cal._assignVal_(this.a, a_next,0, a_dim, t);

            var yt_pred = res["yt_pred"];
            var cache = res["cache"];
            let yt_dim = this.cal.shape(this.cal.transpose(yt_pred));
            this.y_pred = this.cal._assignVal_(this.y_pred, this.cal.transpose(yt_pred), 0, yt_dim, t);

            c_next = res["c_next"];
            var c_dim = this.cal.shape(c_next);
            this.c = this.cal._assignVal_(this.c, c_next, 0, c_dim, t);
            caches.push(cache);
        }
        caches = [caches, this.X];

        return {
            "a": this.a,
            "c": this.c,
            "y_pred": this.y_pred,
            "caches": caches};
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
            xt = caches[8];


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

       var da_prev = this.cal.add(
           this.cal.dot(
               this.cal.transpose(this.cal.sliceMatrix(this.Wf, this.n_x, this.Wf[0].length)),
               this.cal.transpose(dft)
           ),this.cal.add(
               this.cal.dot(
                   this.cal.transpose(this.cal.sliceMatrix(this.Wc, this.n_x, this.Wc[0].length)),
                   this.cal.transpose(dcct)
               ),this.cal.add(
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(this.Wi, this.n_x, this.Wi[0].length)),
                       this.cal.transpose(dit)
                   ),
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(this.Wo, this.n_x, this.Wo[0].length)),
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
       var dxt = this.cal.add(
           this.cal.dot(
               this.cal.transpose(this.cal.sliceMatrix(this.Wf, this.n_a, this.Wf[0].length)),
               this.cal.transpose(dft)
           ), this.cal.add(
               this.cal.dot(
                   this.cal.transpose(this.cal.sliceMatrix(this.Wc, this.n_a, this.Wc[0].length)),
                   this.cal.transpose(dcct)
               ),this.cal.add(
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(this.Wo, this.n_a, this.Wo[0].length)),
                       this.cal.transpose(dot)
                   ),
                   this.cal.dot(
                       this.cal.transpose(this.cal.sliceMatrix(this.Wi, this.n_a, this.Wi[0].length)),
                       this.cal.transpose(dit)
                   ), "+"
               ), "+"
           ), "+"
       );

       return {
           "dxt": dxt,
           "da_prev" : da_prev,
           "dc_prev" : dc_prev,
           "dWf" : dWf,
           "dbf" : this.cal.transpose(dbf),
           "dWi" : dWi,
           "dbi" : this.cal.transpose(dbi),
           "dWc" : dWc,
           "dbc" : this.cal.transpose(dbc),
           "dWo" : dWo,
           "dbo" : this.cal.transpose(dbo)
       }
    }

    lstm_backward(cache){
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
        var caches = cache[0],
            x = cache[1];
        var a1 = caches[0][0],
            c1 = caches[0][1],
            a0 = caches[0][2],
            c0 = caches[0][3],
            f1 = caches[0][4],
            i1 = caches[0][5],
            cc1= caches[0][6],
            o1 = caches[0][7],
            x1 = caches[0][8];

        var dx = this.cal.generateZeros([this.n_x,this.m,this.T_x]),
            da0 = this.cal.generateZeros([this.n_a,this.m]),
            da_prevt = this.cal.generateZeros([this.n_a,this.m]),
            dc_prevt = this.cal.generateZeros([this.n_a,this.m]),
            dWf = this.cal.generateZeros([this.n_a, this.n_a + this.n_x]),
            dWi = this.cal.generateZeros([this.n_a, this.n_a + this.n_x]),
            dWc = this.cal.generateZeros([this.n_a, this.n_a + this.n_x]),
            dWo = this.cal.generateZeros([this.n_a, this.n_a + this.n_x]),
            dbf = this.cal.generateZeros([this.n_a,1]),
            dbi = this.cal.generateZeros([this.n_a,1]),
            dbc = this.cal.generateZeros([this.n_a,1]),
            dbo = this.cal.generateZeros([this.n_a,1]);

        for (var t = this.T_x - 1; t >= 0; t--){
            var gradients = this.lstm_cell_backward(
                this.cal.add(
                    this.cal.getMMinusOneDim(this.da, t),
                    da_prevt, "+"
                ), dc_prevt, caches[t]
            );

            var dxt_dim = this.cal.shape(gradients["dxt"]);
            dx = this.cal._assignVal_(dx, gradients["dxt"], 0, dxt_dim, t);
            dWf = this.cal.add(dWf, gradients["dWf"], "+");
            dWi = this.cal.add(dWi, gradients["dWi"], "+");
            dWc = this.cal.add(dWc, gradients["dWc"], "+");
            dWo = this.cal.add(dWo, gradients["dWo"], "+");
            dbf = this.cal.add(dbf, gradients["dbf"], "+");
            dbi = this.cal.add(dbi, gradients["dbi"], "+");
            dbc = this.cal.add(dbc, gradients["dbc"], "+");
            dbo = this.cal.add(dbo, gradients["dbo"], "+");
        }
        this.Wf = this.cal.add(this.Wf, dWf, "+");
        this.Wi = this.cal.add(this.Wi, dWi, "+");
        this.Wc = this.cal.add(this.Wc, dWc, "+");
        this.Wo = this.cal.add(this.Wo, dWo, "+");
        this.bf = this.cal.add(this.bf, dbf, "+");
        this.bi = this.cal.add(this.bi, dbi, "+");
        this.bc = this.cal.add(this.bc, dbc, "+");
        this.bo = this.cal.add(this.bo, dbo, "+");
        da0 = gradients["da_prev"];
        this.a0 = this.cal.add(this.a0, da0, "+");
        return {
            "dx": dx,
            "da0": da0,
            "dWf": dWf,
            "dbf": dbf,
            "dWi": dWi,
            "dbi": dbi,
            "dWc": dWc,
            "dbc": dbc,
            "dWo": dWo,
            "dbo": dbo
        }

    }

    train(batch){
        for (var i = 0; i < batch; i++){
            var caches = this.lstm_forward()["caches"];
            this.lstm_backward(caches);
        }
    }

    predict(X){
        this.X = X;
        return this.lstm_forward();
    }
}
