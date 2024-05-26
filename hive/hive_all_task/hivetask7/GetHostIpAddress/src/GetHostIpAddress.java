package hw.mipt;

import java.util.ArrayList;
import java.util.List;
import java.util.BitSet;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;

public class GetHostIpAddress extends GenericUDTF {
    public static final int BITS_IN_OCTET = 8;
    public static final int OCTETS_COUNT = 4;
    public static final int BITS_IN_IP = 32;

    private StringObjectInspector ip_inspector;
    private StringObjectInspector mask_inspector;
    private int max_printed = 0;
    private int cur_printed = 0;

    /**
     * Since the processing of 1 record may give us a full table,
     * Hive uses an array for its storage.
     * However, in this example we have a single field in each record.
     * Therefore our array will always have 1 element.
     */
    private Object[] forwardObjArray = new Object[2];

    @Override
    public StructObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {
        if(args.length != 2) {
            throw new UDFArgumentException(getClass().getSimpleName() + " takes only 2 arguments!");
        }

        ip_inspector = (StringObjectInspector) args[0];
        mask_inspector = (StringObjectInspector) args[1];

        // Describing the structure for output.
        // Column names.
        final List<String> fieldNames = new ArrayList<String>() {
            {
                add("host_ips");
                add("value");
            }
        };
        // Inspectors of fields.
        final List<ObjectInspector> fieldInspectors = new ArrayList<ObjectInspector>() {
            {
                add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
                add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
            }
        };
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldInspectors);
    }

    @Override
    public void process(Object[] objects) throws HiveException {
        // UDTF has 1 argument, hence `objects` has a single element too.
        String ip_str = ip_inspector.getPrimitiveJavaObject(objects[0]);
        BitSet ip = IpStrToBitset(ip_str);

        String mask_str = mask_inspector.getPrimitiveJavaObject(objects[0]);
        BitSet mask = IpStrToBitset(mask_str);

        int mask_set = mask.nextSetBit(0);

        max_printed = 1 << mask_set;
        cur_printed = 0;

        RecursivePrintHostIps(ip, mask_set);
    }

    @Override
    public void close() throws HiveException {
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////                      Internals                             ////////
    ////////////////////////////////////////////////////////////////////////////

    static private void ParseOctetToBitset(String num, BitSet bitset, int octet) {
        int val = Integer.parseInt(num);
        int start_i = octet * BITS_IN_OCTET;
        for (int i = 0; i < BITS_IN_OCTET; ++i) {
            bitset.set(start_i + i, (val & 1) == 1);
            val >>= 1;
        }
    }

    static private BitSet IpStrToBitset(String ip_str) {
        final BitSet res = new BitSet(BITS_IN_IP);
        String[] octets = ip_str.split("\\.");

        for (int i = 0; i < OCTETS_COUNT; ++i) {
            ParseOctetToBitset(octets[OCTETS_COUNT - i - 1], res, i);
        }
        return res;
    }

    static private String ParseOctetToStr(BitSet bitset, int octet) {
        int start_i = octet * BITS_IN_OCTET;
        int end_i = start_i + BITS_IN_OCTET - 1;
        int res_int = 0;
        for (int i = end_i; i >= start_i; --i) {
            res_int += (bitset.get(i)) ? 1 : 0;
            res_int <<= 1;
        }
        res_int >>= 1;
        return String.valueOf(res_int);
    }

    static private String IpBitsetToStr(BitSet bitset) {
        String[] octets = new String[4];

        for (int i = 0; i < OCTETS_COUNT; ++i) {
            octets[i] = ParseOctetToStr(bitset, OCTETS_COUNT - i - 1);
        }

        return String.join(".", octets);
    }

    static private String GetValueFromIp(BitSet bitset) {
        long res_long = 0;
        for (int i = BITS_IN_IP - 1; i >= 0; --i) {
            res_long += (bitset.get(i)) ? 1 : 0;
            res_long <<= 1;
        }
        return String.valueOf(res_long >> 1);
    }

    private void RecursivePrintHostIps(BitSet ip, int count) throws HiveException {
        if (count == 0) {
            PrintToTable(ip);
            return;
        }

        ip.clear(count - 1);
        RecursivePrintHostIps(ip, count - 1);
        ip.set(count - 1);
        RecursivePrintHostIps(ip, count - 1);
    }

    private void PrintToTable(BitSet bitset) throws HiveException {
        if (cur_printed == 0 || cur_printed == max_printed - 1) {
            ++cur_printed;
            return;
        }
        ++cur_printed;

        forwardObjArray[0] = IpBitsetToStr(bitset);
        forwardObjArray[1] = GetValueFromIp(bitset);
        forward(forwardObjArray);
    }
}
