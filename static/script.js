async function predictDifficulty() {
  const description = document.getElementById("desc").value;
  const inputDesc = document.getElementById("input_desc").value;
  const outputDesc = document.getElementById("output_desc").value;

  if (!description) {
    alert("Please enter a problem description.");
    return;
  }

  //Prepare the payload
  const payload = {
    description: description,
    input_description: inputDesc,
    output_description: outputDesc,
  };

  // Send to Python Backend
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    // Update UI with results obtained from backend
    const resultBox = document.getElementById("result-box");
    const classEl = document.getElementById("res-class");
    const scoreEl = document.getElementById("res-score");

    classEl.innerText = data.problem_class; 
    scoreEl.innerText = data.problem_score; 

    // Color coding for visual appearance
    if (data.problem_class.toLowerCase() === "easy")
      classEl.style.color = "#28a745";
    else if (data.problem_class.toLowerCase() === "medium")
      classEl.style.color = "#ffc107";
    else classEl.style.color = "#dc3545";

    resultBox.classList.remove("hidden");
  } catch (error) {
    console.error("Error:", error);
    alert("Failed to get prediction. Make sure the server is running.");
  }
}
async function clearFields() {
  document.getElementById("desc").value = "";
  document.getElementById("input_desc").value = "";
  document.getElementById("output_desc").value = "";
  const resultBox = document.getElementById("result-box");
  resultBox.classList.add("hidden");
}
