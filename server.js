const express = require('express');
var path = require('path');
const app = express();
app.set('view engine', 'ejs');
app.get('/', function(req, res) {
    res.render("index",{
    })
});

app.use(express.static(__dirname + '/public'));


app.listen(process.env.PORT || 3000, function(){
  console.log("Express server listening on port %d in %s mode", this.address().port, app.settings.env);
});
