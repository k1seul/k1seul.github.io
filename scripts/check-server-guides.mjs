import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const root = "src/content/server";
const read = (file) => {
    const source = fs.readFileSync(file, "utf8");
    const frontmatter = source.match(/^---\n([\s\S]*?)\n---\n/);
    assert(frontmatter, `${file}: missing frontmatter`);
    const metadata = Object.fromEntries(frontmatter[1].split("\n").map((line) => {
        const separator = line.indexOf(":");
        return [line.slice(0, separator), line.slice(separator + 1).trim().replace(/^"|"$/g, "")];
    }));
    return { source, metadata };
};

const names = fs.readdirSync(root).filter((name) => name.endsWith(".md")).sort();
assert(names.length > 0, "No English guides found");
assert.deepEqual(fs.readdirSync(path.join(root, "ko")).filter((name) => name.endsWith(".md")).sort(), names, "Translation pairs must match");
const routes = new Set(["/server", "/server/gsds", "/ko/server", "/ko/server/gsds"]);
for (const name of names) {
    for (const prefix of ["/server/", "/ko/server/"]) routes.add(prefix + name.replace(/\.md$/, ""));
}

let checkedLinks = 0;
for (const name of names) {
    const english = read(path.join(root, name));
    const korean = read(path.join(root, "ko", name));
    for (const key of ["section", "order"]) assert.equal(korean.metadata[key], english.metadata[key], `${name}: ${key} differs between languages`);
    for (const [language, doc] of [["en", english], ["ko", korean]]) {
        const file = `${language}/${name}`;
        assert.equal(doc.metadata.language, language, `${file}: wrong language metadata`);
        assert(["laal", "gsds", "gsds-specs"].includes(doc.metadata.section), `${file}: invalid guide section`);
        assert(doc.metadata.title && doc.metadata.description, `${file}: missing title or description`);
        assert.equal((doc.source.match(/^```/gm) ?? []).length % 2, 0, `${file}: unclosed code fence`);
        if (language === "en") assert(!/[가-힣]/u.test(doc.source), `${file}: Korean text in the English guide`);
        if (doc.metadata.section === "gsds") {
            assert(!/R730|H730|RAIDZ|18TB|10GbE|iDRAC|백플레인|무상 A\/S/i.test(doc.source), `${file}: hardware details in user guidance`);
        }
        for (const match of doc.source.matchAll(/(?:\]\(|(?:href|src)=")((?:\/ko)?\/server[^\s"\)#]*|\/images\/[^\s"\)#]+)/g)) {
            const target = match[1].replace(/\/$/, "");
            checkedLinks++;
            if (target.startsWith("/images/")) {
                assert(fs.existsSync("public" + target), `${file}: missing image ${target}`);
            } else {
                assert(routes.has(target), `${file}: missing route ${target}`);
                assert.equal(target.startsWith("/ko/"), language === "ko", `${file}: link switches language unexpectedly: ${target}`);
            }
        }
    }
}
console.log(`Checked ${names.length} translation pairs and ${checkedLinks} guide/image links.`);
