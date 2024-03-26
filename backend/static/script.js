// from https://www.codingnepalweb.com/search-bar-autocomplete-search-suggestions-javascript/

// getting all required elements
const searchWrapper = document.querySelector(".search-input");
const inputBox = searchWrapper.querySelector("input");
const suggBox = searchWrapper.querySelector(".autocom-box");
const icon = searchWrapper.querySelector(".icon");
let linkTag = searchWrapper.querySelector("a");
let webLink;
// if user press any key and release
inputBox.onkeyup = (e) => {
  let userData = e.target.value; //user enetered data
  let emptyArray = [];
  if (userData) {
    emptyArray = suggestions.filter((data) => {
      //filtering array value and user characters to lowercase and return only those words which are start with user enetered chars
      return data.toLocaleLowerCase().includes(userData.toLocaleLowerCase());
    });
    emptyArray = emptyArray.map((data) => {
      // passing return data inside li tag
      return (data = `<li>${data}</li>`);
    });
    searchWrapper.classList.add('active'); //show autocomplete box
    showSuggestions(emptyArray);
    let allList = suggBox.querySelectorAll('li');
    for (let i = 0; i < allList.length; i++) {
      //adding onclick attribute in all li tag
      allList[i].setAttribute('onclick', 'select(this)');
    }
  } else {
    searchWrapper.classList.remove('active'); //hide autocomplete box
  }
};

function select(element) {
  let selectData = element.textContent;
  inputBox.value = selectData;
  icon.onclick = () => {
    webLink = `https://www.google.com/search?q=${selectData}`;
    linkTag.setAttribute("href", webLink);
    linkTag.click();
  }
  searchWrapper.classList.remove("active");
}

function showSuggestions(list) {
  let listData;
  if (!list.length) {
    userValue = inputBox.value;
    listData = `<li>${userValue}</li>`;
  } else {
    listData = list.join('');
  }
  suggBox.innerHTML = listData;
}

//Testing select2 -> ignore
$(document).ready(function () {
  $("#filter-text-val").select2({
    data: albums.map((album) => ({ id: album, text: album })),
  });

  $("#filter-text-val").on("select2:select", function (e) {
    const selectedAlbum = e.params.data.text;
    inputBox.value = selectedAlbum;
  });
});

function filterText() {
  document.getElementById("answer-box").innerHTML = "";
  const albumValue = $("#filter-text-val").val();
  const genreValue = document.getElementById("filter-text-genre").value;
  const composerValue = document.getElementById("filter-text-composer").value;

  fetch(
    "/albums?" +
    new URLSearchParams({
      title: albumValue,
      genre: genreValue,
      composer: composerValue,
    }).toString()
  )
    .then((response) => response.json())
    .then((data) =>
      data.forEach((row) => {
        let tempDiv = document.createElement("div");
        tempDiv.innerHTML = answerBoxTemplate(row.albums, row.reviews);
        document.getElementById("answer-box").appendChild(tempDiv);
      })
    );
}
//Testing select2 -> ignore