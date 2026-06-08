package common.projectparser;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import common.utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;

public class ConstructorExtr extends VoidVisitorAdapter<Void> {

    private final ArrayList<HashMap<String, Object>> constructorInfoHashArray;

    public ConstructorExtr() {
        this.constructorInfoHashArray = new ArrayList<>();
    }

    public ArrayList<HashMap<String, Object>> getConstructorInfo(CompilationUnit compilationUnit) {
        compilationUnit.accept(new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(ConstructorDeclaration cd, Void arg) {
                HashMap<String, Object> constructorInfo = new HashMap<>();

                // CM
                constructorInfo.put("CONSTRUCTORMODIFIER", Utils.nodeListToArrayList(cd.getModifiers()));

                // CP
                constructorInfo.put("CONSTRUCTORPARAMETER", Utils.getParameters(cd.getParameters()));

                constructorInfoHashArray.add(constructorInfo);
            }
        }, null);

        return constructorInfoHashArray;
    }
}
