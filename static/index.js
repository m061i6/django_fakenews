$(function(){
    console.log('js start!');
    $('.result').hide();
    $('#again').on('click',rePresict);
});
function rePresict(name) {
    $("#inputText").val('');
    $("#inputText").focus();
    $('html,body').animate({ scrollTop: 0 }, 'slow');
}
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
function predict(){
    if( $("#inputText").val()=="" || !$.trim($("#inputText").val())){    
        alert("請輸入欲判斷的文字內容");
        $("#inputText").val('');
        $("#inputText").focus();
        return false;
    }
    let input_text = $('#inputText').val();
    
    $('.result').show();
    $('#title').html('模型預測中...')
    $('.loading').show();
    $('.finish').hide(); 
    startpredict(input_text)
    // showResult(input_text);
    $('#yourtext').html(input_text);
    $([document.documentElement, document.body]).animate({
        scrollTop: $("#title").offset().top
    }, 1000);
    return false;
}
function startpredict(input_text){
    let form_data  = new FormData();
    form_data.append('input_text', input_text);
    // for (var key of form_data.entries()) {
    //     console.log(key[0] + ', ' + key[1]);
    // }
    fetch('./predict', {
            method: 'POST',
            credentials: "same-origin",
            headers: {
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: form_data,
        })
        .then(res => res.json())
        .catch(error => console.error(error))
        .then(result  => {
            res = result;
            console.log(res);
            showResult(res);
        })
}
function showResult(res){
    $('.result').show();
    $('.loading').hide();
    $('.finish').show();
    let label=''
    if(res.result){
        label = '<span class="btn-success">真</span>';
    }
    else{
        label = '<span class="btn-danger">假</span>';
    }
    let title = '判斷結果為'+label;
    let info = '信心指數'+Number(res.confidence*100).toFixed(2)+'%'
    // let info = '信心指數 78.92283916%'
    $('#title').html(title);
    $('#info').html(info);
}
