package common.projectparser;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import common.utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;

public class VariableExtr extends VoidVisitorAdapter<Void> {
    private final ArrayList<HashMap<String, Object>> variableInfoHashArray;

    public VariableExtr() {
        this.variableInfoHashArray = new ArrayList<>();
    }

    public ArrayList<HashMap<String, Object>> getVariableInfo(CompilationUnit compilationUnit) {
        compilationUnit.accept(new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(VariableDeclarationExpr vde, Void arg) {
                HashMap<String, Object> variableInfo = new HashMap<>();

                variableInfo.put("VARIABLEDECLARATION", Utils.nodeListToArrayList(vde.getVariables()));
                variableInfo.put("VARIABLEELEMENTTYPE", vde.getElementType().toString());
                variableInfo.put("VARIABLECOMMONTYPE", vde.getCommonType().toString());

                variableInfoHashArray.add(variableInfo);
            }
        }, null);

        return variableInfoHashArray;
    }

}
