// runs for all .drop-zone__input on page ie 1+ dropzone box
document.querySelectorAll(".drop-zone__input").forEach((inputElement) => {
  const dropZoneElement = inputElement.closest(".drop-zone"); // selects enclosing div class="drop-zone"
  
  dropZoneElement.addEventListener("click", (e) => {
    inputElement.click();
  });

  inputElement.addEventListener("change", (e) => {
    if (inputElement.files.length) {
      updateThumbnail(dropZoneElement, inputElement.files[0]);
    }
  });

  dropZoneElement.addEventListener("dragover", (e) => { // runs when img is dragged over box
    e.preventDefault();
    dropZoneElement.classList.add("drop-zone--over"); // change border style from dash to solid
  });

  ["dragleave", "dragend"].forEach((type) => { // handle either of these events are applied 
    // dragleave says when img dragged back outside box
    //  dragend says when ESC pressed, canceling upload
    dropZoneElement.addEventListener(type, (e) => { // then grab hold of specified event obj 
      dropZoneElement.classList.remove("drop-zone--over"); // goes back to dash border from solid
    });
  });
  
  dropZoneElement.addEventListener("drop", (e) => {
    e.preventDefault(); // stop from showing image on page

    if (e.dataTransfer.files.length) {
      inputElement.files = e.dataTransfer.files;
      updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
    }

    dropZoneElement.classList.remove("drop-zone--over");
  });
});

/**
 * Updates the thumbnail on a drop zone element.
 *
 * @param {HTMLElement} dropZoneElement
 * @param {File} file
 */
function updateThumbnail(dropZoneElement, file) {
  let thumbnailElement = dropZoneElement.querySelector(".drop-zone__thumb");

  // First time - remove the prompt
  if (dropZoneElement.querySelector(".drop-zone__prompt")) {
    dropZoneElement.querySelector(".drop-zone__prompt").remove();
  }

  // First time - there is no thumbnail element, so lets create it
  if (!thumbnailElement) {
    thumbnailElement = document.createElement("div");
    thumbnailElement.classList.add("drop-zone__thumb");
    dropZoneElement.appendChild(thumbnailElement);
  }

  thumbnailElement.dataset.label = file.name;

  // Show thumbnail for image files
  const reader = new FileReader();

  if (file.type.startsWith("image/tiff")) {
    thumbnailElement.style.backgroundImage = null;
    window.alert("TIFF display not supported in Browser only JPG/PNG etc.")
    reader.readAsDataURL(file);
    reader.onload = () => {
      thumbnailElement.style.backgroundImage = `url('${reader.result}')`;
    };

  } else if (file.type.startsWith("image/")) {

    reader.readAsDataURL(file);
    reader.onload = () => {
      thumbnailElement.style.backgroundImage = `url('${reader.result}')`;
      // submit to /upload_image
      
    };
  } else {
    thumbnailElement.style.backgroundImage = null;
  }
}
