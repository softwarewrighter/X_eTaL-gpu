// xetal doc: search by name or by type over the items of the documented
// files and the built-ins (search-index.js). A query with an arrow, a
// "=>", or starting with a capital is a type, compared Hoogle-like: its
// constraints dropped and its type variables renamed a, b, ... in order
// (as the index's normalized types are), so "Num a => a -> a -> a" finds
// +, m_ax and the like, and "Char -> Char" finds every function from
// text to text (or whose type holds that). Any other query is a name,
// matched without prefix or underline: "mean" finds l:m_ean.
(function () {
  function norm(t) {
    var i = t.indexOf("=>");
    if (i >= 0) t = t.slice(i + 2);
    var toks = t.match(/[A-Za-z0-9_]+|[^A-Za-z0-9_\s]+/g) || [];
    var seen = [];
    return toks.map(function (x) {
      if (!/^[a-z]/.test(x)) return x;
      var at = seen.indexOf(x);
      if (at < 0) { seen.push(x); at = seen.length - 1; }
      return String.fromCharCode(97 + (at % 26));
    }).join(" ");
  }
  function key(name) {
    var at = name.lastIndexOf(":");
    return name.slice(at + 1).replace(/[^A-Za-z0-9]/g, "").toLowerCase();
  }
  function isType(q) {
    return q.indexOf("->") >= 0 || q.indexOf("=>") >= 0 || /^[A-Z]/.test(q);
  }
  function score(row, q, type) {
    if (type) {
      var nq = norm(q);
      if (row[3] === nq) return 0;
      return (" " + row[3] + " ").indexOf(" " + nq + " ") >= 0 ? 1 : -1;
    }
    var k = key(row[0]), kq = key(q);
    if (row[0] === q || (kq && k === kq)) return 0;
    if (!kq) return row[0].indexOf(q) >= 0 ? 2 : -1;
    if (k.indexOf(kq) === 0) return 1;
    return k.indexOf(kq) >= 0 || row[0].indexOf(q) >= 0 ? 2 : -1;
  }
  function find(q) {
    var type = isType(q), found = [];
    (window.XETAL_DOC_INDEX || []).forEach(function (row) {
      var s = score(row, q, type);
      if (s >= 0) found.push([s, row]);
    });
    found.sort(function (a, b) {
      return a[0] - b[0] || (a[1][0] < b[1][0] ? -1 : a[1][0] > b[1][0] ? 1 : 0);
    });
    return found.slice(0, 60).map(function (f) { return f[1]; });
  }
  function cell(text, tag) {
    var td = document.createElement(tag || "td");
    td.textContent = text;
    return td;
  }
  function show(results, rows) {
    results.textContent = "";
    var table = document.createElement("table");
    table.className = "items";
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      var name = document.createElement("td");
      var a = document.createElement("a");
      a.href = row[5];
      a.textContent = row[0];
      name.appendChild(a);
      tr.appendChild(name);
      tr.appendChild(cell(row[1]));
      var ty = cell(row[2]);
      ty.className = "type";
      tr.appendChild(ty);
      tr.appendChild(cell(row[4] ? row[4] + ": " + row[6] : row[6]));
      table.appendChild(tr);
    });
    var head = document.createElement("h2");
    head.textContent = rows.length ? "Results" : "Nothing found";
    results.appendChild(head);
    results.appendChild(table);
  }
  window.addEventListener("DOMContentLoaded", function () {
    var input = document.getElementById("search");
    var results = document.getElementById("results");
    if (!input || !results) return;
    input.addEventListener("input", function () {
      var q = input.value.trim();
      document.body.classList.toggle("searching", q.length > 0);
      if (q) show(results, find(q));
    });
  });
})();
