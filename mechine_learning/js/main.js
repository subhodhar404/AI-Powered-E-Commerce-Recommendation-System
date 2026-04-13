(function ($) {
    "use strict";

    // mobile menu 
    $('#mobile-menu').meanmenu({
        meanScreenWidth: "991",
        meanMenuContainer: '.mobile-menu'
    });

    // icon show js
    const loginIcon = document.getElementById('loginIcon');
    const registerIcon = document.getElementById('registerIcon');

    document.getElementById('home-tab')?.addEventListener('shown.bs.tab', function () {
        loginIcon?.classList.remove('d-none');
        registerIcon?.classList.add('d-none');
    });

    document.getElementById('profile-tab')?.addEventListener('shown.bs.tab', function () {
        registerIcon?.classList.remove('d-none');
        loginIcon?.classList.add('d-none');
    });

    // niceselect js 
    $(document).ready(function() {
        $('select').niceSelect();
    });

    // zoom image js
    $('.zoom').zoom({
        magnify: 2
    });

})(jQuery);