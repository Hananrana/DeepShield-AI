from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag in ['img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'col', 'embed', 'param', 'track', 'wbr']:
            return
        self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in ['img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'col', 'embed', 'param', 'track', 'wbr']:
            return
        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag}> at line {self.getpos()[0]}")
            return
        
        expected_tag, pos = self.stack.pop()
        if expected_tag != tag:
            self.errors.append(f"Mismatched tag at line {self.getpos()[0]}: expected </{expected_tag}> (opened at line {pos[0]}), got </{tag}>.")
            # Try to recover
            for i in range(len(self.stack)-1, -1, -1):
                if self.stack[i][0] == tag:
                    self.stack = self.stack[:i]
                    break

parser = MyHTMLParser()
with open(r"c:\Users\hanan\Desktop\DeepShield\DeepShield\templates\index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

parser.feed(html_content)
if parser.stack:
    print("Unclosed tags remaining on stack:")
    for tag, pos in reversed(parser.stack):
        print(f"  <{tag}> opened at line {pos[0]}")
else:
    print("No unclosed tags!")

if parser.errors:
    print("\nErrors found:")
    for err in parser.errors:
        print(" ", err)
