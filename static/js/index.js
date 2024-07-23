let slideIndex = 0;
let slider = document.getElementById('slider')
let slides = slider.getElementsByClassName('slide')
setTimeout(() => {
	slides[slideIndex].classList.add('active')
}, 500)