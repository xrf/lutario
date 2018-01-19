(function() {var implementors = {};
implementors["bincode"] = ["impl&lt;'de, 'a, R, S, E&gt; <a class=\"trait\" href=\"serde/de/trait.Deserializer.html\" title=\"trait serde::de::Deserializer\">Deserializer</a>&lt;'de&gt; for &amp;'a mut <a class=\"struct\" href=\"bincode/internal/struct.Deserializer.html\" title=\"struct bincode::internal::Deserializer\">Deserializer</a>&lt;R, S, E&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;R: <a class=\"trait\" href=\"bincode/read_types/trait.BincodeRead.html\" title=\"trait bincode::read_types::BincodeRead\">BincodeRead</a>&lt;'de&gt;,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"bincode/trait.SizeLimit.html\" title=\"trait bincode::SizeLimit\">SizeLimit</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;E: <a class=\"trait\" href=\"byteorder/trait.ByteOrder.html\" title=\"trait byteorder::ByteOrder\">ByteOrder</a>,&nbsp;</span>",];
implementors["serde_yaml"] = ["impl&lt;'de&gt; <a class=\"trait\" href=\"serde/de/trait.Deserializer.html\" title=\"trait serde::de::Deserializer\">Deserializer</a>&lt;'de&gt; for <a class=\"enum\" href=\"serde_yaml/enum.Value.html\" title=\"enum serde_yaml::Value\">Value</a>","impl&lt;'de&gt; <a class=\"trait\" href=\"serde/de/trait.Deserializer.html\" title=\"trait serde::de::Deserializer\">Deserializer</a>&lt;'de&gt; for <a class=\"struct\" href=\"serde_yaml/struct.Number.html\" title=\"struct serde_yaml::Number\">Number</a>","impl&lt;'de, 'a&gt; <a class=\"trait\" href=\"serde/de/trait.Deserializer.html\" title=\"trait serde::de::Deserializer\">Deserializer</a>&lt;'de&gt; for &amp;'a <a class=\"struct\" href=\"serde_yaml/struct.Number.html\" title=\"struct serde_yaml::Number\">Number</a>",];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        
})()
