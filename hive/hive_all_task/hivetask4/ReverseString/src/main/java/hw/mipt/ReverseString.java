package hw.mipt;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

@Description(
        name = "flipper",
        value = "Returns reversed string.",
        extended = "Example:\n" +
                   "SELECT reverse_string(field) from a;"
)
public class ReverseString extends UDF {

    public String evaluate(String str) {
        StringBuilder sb = new StringBuilder(str);
        sb.reverse();
        return sb.toString();
    }
}
